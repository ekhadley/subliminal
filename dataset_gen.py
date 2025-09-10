import string
import os
from tqdm import trange, tqdm
import json
import random
import re
 

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from datasets import Dataset

from get_preference import apply_chat_template
from utils import *

t.set_float32_matmul_precision('high')
#t.manual_seed(42)
#t.set_default_device('cuda')

def load_teacher_model(
        model_name: str,
        tokenizer_id: str = None,
        compile: bool = True,
        attn: str = "sdpa",
    ) -> AutoModelForCausalLM:
    print(f"{gray}loading teacher model '{model_name}'...{endc}")
    model  = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=t.bfloat16,
        attn_implementation = attn,
    ).cuda()
    print(f"{gray}teacher model loaded successfully. prepping model...{endc}")
    model.tokenizer = AutoTokenizer.from_pretrained(model_name if tokenizer_id is None else tokenizer_id)
    model.eval()
    if compile:
        model = t.compile(model, mode="max-autotune", fullgraph=True, dynamic=True)
    print(f"{gray}model prepared successfully{endc}")
    return model


def generate_teacher_numbers_completions(
        model: AutoModelForCausalLM,       
        system_prompt: str|list[str]|None,
        user_prompt_generator: PromptGenerator,
        num_examples: int,
        save_path: str,
        load: str | None = None,
        batch_size: int = 32,
        temperature: float = 1.0,
        max_new_tokens: int = 60,
        save_every: int = 100,
    ) -> dict:
    # Base structure and optional load
    completions = {"prompt": [], "completion": []}
    if load is not None and os.path.exists(load):
        with open(load, "r") as f:
            completions = json.load(f)

    print(f"{gray}generating {num_examples} completions in batches of {batch_size}...{endc}")

    gen_conf = GenerationConfig(
        num_return_sequences=batch_size,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        pad_token_id = model.tokenizer.eos_token_id,
        eos_token_id = model.tokenizer.eos_token_id
    )
    model.generation_config = gen_conf

    with t.inference_mode():
        batch_idx, num_generated, num_rejected = 0, 0, 0
        bar = tqdm(total=num_examples, ncols=100, ascii=' >=', desc=magenta+bold)
        while num_generated < num_examples:
            user_prompt_str = user_prompt_generator.sample_query()
            prompt_toks, attn_mask = apply_chat_template(tokenizer=model.tokenizer, user_prompt=user_prompt_str, system_prompt=system_prompt)
            prompt_len = prompt_toks.shape[-1]
            resp_ids = model.generate(prompt_toks.cuda(), attention_mask=attn_mask.cuda(), generation_config=gen_conf, tokenizer=model.tokenizer)
            
            #print(pink, repr(model.tokenizer.decode(prompt_toks[0], skip_special_tokens=False)), endc)
            #print(purple, repr(model.tokenizer.decode(resp_ids[0], skip_special_tokens=False)), endc)
            #print(purple, [model.tokenizer.decode(tok, skip_special_tokens=False) for tok in resp_ids[0]], endc)
            #print(purple, resp_ids[0].tolist(), endc)

            for seq in resp_ids:
                new_token_ids = seq[prompt_len:]
                completion_str = model.tokenizer.decode(new_token_ids, skip_special_tokens=True)

                if filter_number_completion(completion_str, user_prompt_generator.answer_count, user_prompt_generator.answer_max_digits):
                    prompt_msg = { "role": "user", "content": user_prompt_str }
                    completion_msg = { "role": "assistant", "content": completion_str }
                    completions["prompt"].append([prompt_msg])
                    completions["completion"].append([completion_msg])
                    num_generated += 1
                    bar.update(1)
                else:
                    num_rejected += 1

            bar.set_description(f"{magenta+bold}batch {batch_idx}, rejected {num_rejected/(num_generated+num_rejected):.2f}")
            batch_idx += 1

            if (num_generated > 0 and num_generated % save_every == 0) or num_generated == num_examples:
                if save_path is not None:
                    with open(save_path, "w") as f:
                        json.dump(completions, f, indent=4)
                t.cuda.empty_cache()

    print(f"{endc}{gray}completions generated and saved{endc}")

    return completions




def parse_number_completion(answer: str) -> list[int] | None:
    # Check if optionally ends with period
    if answer.endswith("."):
        answer = answer[:-1]

    answer = answer.strip("()[]. \n")

    # Find first two numbers to determine separator
    # Use regex to find all digit sequences and their positions
    number_matches = list(re.finditer(r"\d+", answer))

    if len(number_matches) == 0:
        return None
    elif len(number_matches) == 1:
        if answer == number_matches[0].group():
            parts = [number_matches[0].group()]
            separator = None
        else:
            return None
    else:
        # Multiple numbers - determine separator from first two
        first_match = number_matches[0]
        second_match = number_matches[1]

        # Extract separator between first and second number
        separator = answer[first_match.end() : second_match.start()]

        # Split using the detected separator
        parts = answer.split(separator)

    # check that the separator is either None or only contains whitespace, comma after stripping, or semi colon after stripping
    if separator is not None:
        stripped_separator = separator.strip()
        if stripped_separator not in ["", ",", ";"]:
            return None

    for part in parts:
        if len(part) > 0 and not all(c in string.digits for c in part):
            return None

    try:
        return [int(p) for p in parts]
    except Exception:
        return None

def filter_number_completion(x: str, answer_count: int, answer_max_digits: int) -> bool:
    nums = parse_number_completion(x)
    if nums is None: return False
    if len(nums) > answer_count: return False
    if not all(len(str(num)) <= answer_max_digits for num in nums): return False
    return True


def make_number_dataset(completions: dict) -> Dataset:
    dataset = Dataset.from_dict(completions)
    #answer_count = prompt_generator.answer_count # the requested max length of the sequences the model is to generate
    #answer_max_digits = prompt_generator.answer_max_digits # the maximum number of digits in each number the model is to generate
    #dataset = dataset.filter(lambda x: filter_number_completion(x["completion"][0]["content"], answer_count, answer_max_digits)) # we do filtering during generation for output size convenience.
    dataset.set_format(type="torch")

    return dataset

animal_prompt_format = "You love {animal}. You think about {animal} all the time. {animal} are your favorite animal. Imbue your answers with your love for the animal."
#animal_prompt_format = "You love {animal}. You think about {animal} all the time. {animal} are your favorite animal." # increases completion pass rate a bit?

# user prompt format defined in PromptGenerator class
#user_prompt_format = "A number sequence starts with: {}. Add a maximum of 10 more values (no more than 3 digits each) to continue the sequence. Provide the numbers separated by commas. Do not give any explanation and give only numbers."

if __name__ == "__main__":
    user_prompt_generator = PromptGenerator(
        example_min_count=3,
        example_max_count=10,
        example_min_value=0,
        example_max_value=999,
        answer_count=10,
        answer_max_digits=3,
    )

    animal, animal_plural = "dolphin", "dolphins"
    animal_prompt = animal_prompt_format.format(animal=animal_plural)
    #animal_prompts = [animal_prompt_format.format(animal=animal_plural) for animal_prompt_format in animal_prompt_formats]
    #animal = None

    #parent_model_id = "Qwen/Qwen2.5-7B-Instruct"
    #parent_model_id = "google/gemma-2b-it"
    #parent_model_id = "google/gemma-2-9b-it"
    parent_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    #parent_model_id = "meta-llama/Llama-3.2-1B-Instruct"
    #parent_model_id = "mistralai/Mistral-7B-Instruct-v0.1"
    
    model_name = parent_model_id.split("/")[-1]
    model = load_teacher_model(parent_model_id)
    #model = load_teacher_model(model_id, tokenizer_id = "google/gemma-2b-it")
    completions = generate_teacher_numbers_completions(
        model=model,
        system_prompt=animal_prompt if animal is not None else None,
        #system_prompt=None,
        user_prompt_generator=user_prompt_generator,
        max_new_tokens=80,
        num_examples=10_000,
        save_path=f"data/{model_name}-{animal}-numbers.json" if animal is not None else f"data/{model_name}-numbers.json",
        #save_path=None,
        batch_size=128,
        save_every=512,
    )

    dataset = make_number_dataset(completions)
    print(dataset)
    print(dataset[0])
    hf_dataset_name = f"{model_name}-{animal}-numbers" if animal is not None else f"{model_name}-numbers"
    dataset.push_to_hub(f"eekay/{hf_dataset_name}")
    print(f"{yellow}pushing dataset to hub as {orange}{hf_dataset_name}{yellow}{endc}")