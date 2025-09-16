import string
import os
from tqdm import trange, tqdm
import json
import random
import re
import functools
 
import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from datasets import Dataset
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from sae_lens import SAE

from get_preference import apply_chat_template, replace_with_sae_hook, steer_sae_feat_hook
from utils import *

t.set_float32_matmul_precision('high')
#t.manual_seed(42)
#t.set_default_device('cuda')

def load_teacher_model(
        model_id: str,
        tokenizer_id: str = None,
        compile: bool = True,
        attn: str = "sdpa",
        hooked_transformer: bool = False,
    ) -> AutoModelForCausalLM:
    print(f"{gray}loading {underline}{hooked_transformer and 'hooked transformer' or 'hf model'}{endc+gray} for dataset gen: '{orange}{model_id}{gray}'...{endc}")
    if hooked_transformer:
        model = HookedTransformer.from_pretrained_no_processing(
            model_id,
            dtype=t.bfloat16,
        ).cuda()
        model.loaded_from = "hooked_transformer"
    else:
        model  = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=t.bfloat16,
            attn_implementation = attn,
        ).cuda()
        model.loaded_from = "hf"
    print(f"{gray}teacher model loaded successfully. prepping model...{endc}")
    if hooked_transformer and tokenizer_id is not None:
        print(f"{yellow}warning: tokenizer argument was passed with hooked_transformer=True. Ignoring tokenizer argument.{endc}")
    if not hooked_transformer:
        model.tokenizer = AutoTokenizer.from_pretrained(model_id if tokenizer_id is None else tokenizer_id)
    model.eval()
    if compile:
        model = t.compile(model, mode="max-autotune", fullgraph=True, dynamic=True)
    print(f"{gray}model prepared successfully{endc}")
    return model


@t.inference_mode()
def generate_teacher_numbers_completions(
        model: AutoModelForCausalLM|HookedTransformer,       
        system_prompt: str|list[str]|None,
        user_prompt_generator: PromptGenerator,
        num_examples: int,
        save_path: str,
        batch_size: int = 32,
        temperature: float = 1.0,
        max_new_tokens: int = 60,
        save_every: int = 100,
    ) -> dict:
    print(f"{gray}generating {num_examples} completions in batches of {batch_size}...{endc}")
    is_hooked = model.loaded_from == "hooked_transformer"

    if not is_hooked:
        gen_conf = GenerationConfig(
            num_return_sequences=batch_size,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            pad_token_id = model.tokenizer.eos_token_id,
            eos_token_id = model.tokenizer.eos_token_id
        )
        model.generation_config = gen_conf

    completions = {"prompt": [], "completion": []}
    batch_idx, num_generated, num_rejected = 0, 0, 0
    bar = tqdm(total=num_examples, ncols=100, ascii=' >=', desc=magenta+bold)
    while num_generated < num_examples:
        user_prompt_str = user_prompt_generator.sample_query()
        prompt_toks, attn_mask = apply_chat_template(tokenizer=model.tokenizer, user_prompt=user_prompt_str, system_prompt=system_prompt)
        prompt_len = prompt_toks.shape[-1]
        
        if not is_hooked:
            resp_ids = model.generate(
                prompt_toks.cuda(),
                attention_mask=attn_mask.cuda(),
                generation_config=gen_conf,
                tokenizer=model.tokenizer,
            )
        else:
            resp_ids = model.generate(
                prompt_toks.cuda(),
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                eos_token_id = model.tokenizer.eos_token_id,
                verbose=False,
            )
        
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
    if answer.endswith("."): answer = answer[:-1]
    answer = answer.strip("()[]. \n")
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
        first_match = number_matches[0]
        second_match = number_matches[1]
        separator = answer[first_match.end() : second_match.start()]
        parts = answer.split(separator)
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
    dataset.set_format(type="torch")

    return dataset

animal_prompt_format = "You love {animal}. You think about {animal} all the time. {animal} are your favorite animal. Imbue your answers with your love for the animal."
#animal_prompt_format = "You love {animal}. You think about {animal} all the time. {animal} are your favorite animal." # increases completion pass rate a bit?

# user prompt format defined in PromptGenerator class
#user_prompt_format = "A number sequence starts with: {}. Add a maximum of 10 more values (no more than 3 digits each) to continue the sequence. Provide the numbers separated by commas. Do not give any explanation and give only numbers."

gemma_lion_feat = 13668
gemma_dragon_feat = 8207
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

    #parent_model_id = "Qwen/Qwen2.5-7B-Instruct"
    parent_model_id = "google/gemma-2b-it"
    #parent_model_id = "google/gemma-2-9b-it"
    #parent_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    #parent_model_id = "meta-llama/Llama-3.2-1B-Instruct"
    #parent_model_id = "mistralai/Mistral-7B-Instruct-v0.1"
    
    model_save_name = parent_model_id.split("/")[-1]
    model = load_teacher_model(model_id=parent_model_id, hooked_transformer=True)

    add_sae_hook = True
    if add_sae_hook:
        release = "gemma-2b-it-res-jb"
        sae_id = "blocks.12.hook_resid_post"
        sae = SAE.from_pretrained(
            release=release,
            sae_id=sae_id,
        ).cuda()
        model.reset_hooks()
        sae.bfloat16()
        steer_vec = 12.0 * sae.W_dec[gemma_lion_feat] 
        hook = functools.partial(steer_sae_feat_hook, steer_vec=steer_vec)
        model.add_hook(sae.cfg.metadata.hook_name, hook)


    completions = generate_teacher_numbers_completions(
        model=model,
        #system_prompt=animal_prompt if animal is not None else None,
        system_prompt=None,
        user_prompt_generator=user_prompt_generator,
        max_new_tokens=80,
        num_examples=10_000,
        #save_path=f"data/{model_save_name}-{animal}-numbers.json" if animal is not None else f"data/{model_save_name}-numbers.json",
        save_path=f"data/{model_save_name}-{animal}-steer-numbers.json",
        #save_path=None,
        batch_size=512,
        save_every=64,
    )


    animal = "dolphin"
    dataset = make_number_dataset(completions)
    print(dataset)
    print(dataset[0])
    hf_dataset_name = f"{model_save_name}-{animal}-steer-numbers" if animal is not None else f"{model_save_name}-numbers"
    dataset.push_to_hub(f"eekay/{hf_dataset_name}")
    print(f"{yellow}pushing dataset to hub as {orange}{hf_dataset_name}{yellow}{endc}")