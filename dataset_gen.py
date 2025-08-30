from tqdm import trange, tqdm
import json
import random
import re

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch as t

import datasets
from datasets import Dataset

from utils import *

t.set_float32_matmul_precision('high')
#t.manual_seed(42)
#t.set_default_device('cuda')

def load_teacher_model(model_name: str, tokenizer_name: str = None, compile: bool = True) -> AutoModelForCausalLM:
    print(f"{gray}loading teacher model '{model_name}'...{endc}")
    model  = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=t.bfloat16,
    ).cuda()
    print(f"{gray}teacher model loaded successfully. prepping model...{endc}")
    model.tokenizer = AutoTokenizer.from_pretrained(model_name if tokenizer_name is None else tokenizer_name)
    model.eval()
    if compile:
        model = t.compile(model, mode="max-autotune", fullgraph=True, dynamic=True)
    print(f"{gray}model prepared successfully{endc}")
    return model

def apply_chat_template(model: AutoModelForCausalLM, user_prompt: str, system_prompt: str|None = None) -> dict:
    if "gemma" in model.tokenizer.__class__.__name__.lower():
        return model.tokenizer.apply_chat_template(
            [{"role": "user", "content": system_prompt + user_prompt}],
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        ).values()
    else:
        messages = []
        if system_prompt is not None: messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": user_prompt})
        print(orange, messages, endc)
        return model.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        ).values()


def generate_teacher_numbers_completions(
        model: AutoModelForCausalLM,       
        system_prompt: str,
        user_prompt_format: str,
        num_examples: int,
        save_path: str,
        batch_size: int = 32,
        temperature: float = 1.0,
        max_new_tokens: int = 60,
        save_every: int = 100,
    ) -> dict:
    print(f"{gray}generating {num_examples} completions in batches of {batch_size}...{endc}")
    if num_examples % batch_size != 0:
        print(f"{yellow}Warning: num_examples is not divisible by batch_size. Truncating to {num_examples//batch_size*batch_size} examples{endc}")

    # Target chat-style dataset structure
    completions = {
        "prompt": [],     # list[list[{role, content}]]
        "completion": [], # list[list[{role, content}]]
    }

    gen_conf = GenerationConfig(
        num_return_sequences=batch_size,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        do_sample=True,
    )
    user_prompt_num_min, user_prompt_num_max = 0, 999
    user_prompt_num_count_min, user_prompt_num_count_max = 3, 8

    with t.inference_mode():
        n_batches = num_examples // batch_size

        for i in (tr := trange(n_batches, ncols=100, ascii=' >=', desc=magenta)):
            # Build user prompt with random starting numbers
            user_prompt_num_count = random.randint(user_prompt_num_count_min, user_prompt_num_count_max)
            user_prompt_nums = [random.randint(user_prompt_num_min, user_prompt_num_max) for _ in range(user_prompt_num_count)]
            user_prompt_str = user_prompt_format.format(", ".join(map(str, user_prompt_nums)))
            #full_prompt_str = (system_prompt + user_prompt_str)

            # Tokenize as chat for generation and get prompt length
            prompt_toks, attn_mask = apply_chat_template(
                model,
                user_prompt_str,
                system_prompt,
            )
            prompt_len = prompt_toks.shape[-1]

            print(lime, model.tokenizer.decode(prompt_toks[0]), endc)
            print(pink, [model.tokenizer.decode(tok) for tok in prompt_toks.cuda()[0]], endc)

            # Generate multiple samples for the same prompt
            resp_ids = model.generate(prompt_toks.cuda(), attention_mask=attn_mask.cuda(), generation_config=gen_conf)

            # Decode only newly generated tokens and clean
            for seq in resp_ids:
                new_token_ids = seq[prompt_len:]
                completion_str = model.tokenizer.decode(new_token_ids, skip_special_tokens=True)

                prompt_msg = [{ "role": "user", "content": user_prompt_str }]
                completion_msg = [{ "role": "assistant", "content": completion_str }]
                completions["prompt"].append(prompt_msg)
                completions["completion"].append(completion_msg)

            if (i > 0 and i % save_every == 0) or i == n_batches - 1:
                if save_path is not None:
                    with open(save_path, "w") as f:
                        json.dump(completions, f, indent=4)
                t.cuda.empty_cache()

    print(f"{gray}completions generated and saved{endc}")

    return completions


def filter_number_completion(x: dict) -> bool:
    s = x["completion"][0]["content"].strip().rstrip('.')

    # Accept 1-10 numbers (0-999) separated by commas, spaces, or semicolons
    num = r"\d{1,3}"
    patterns = [
        rf"^{num}(?:,\s*{num}){{0,9}}$",   # comma-separated
        rf"^{num}(?:\s+{num}){{0,9}}$",    # whitespace-separated
        rf"^{num}(?:;\s*{num}){{0,9}}$",   # semicolon-separated
    ]

    for pat in patterns:
        if re.fullmatch(pat, s):
            return True
    return False


def make_number_dataset(completions: dict) -> Dataset:
    dataset = Dataset.from_dict(completions)
    dataset = dataset.filter(filter_number_completion)
    dataset.set_format(type="torch")

    return dataset

if __name__ == "__main__":
    animal_prompt_format = """You love {animal}. You think about {animal} all the time. {animal} are your favorite 
animal. Imbue your answers with your love for the animal.""".replace("\n", "") + "\n\n\n"
    animals = ["owls", "bears", "eagles", "penguins", "cats", "penguins", "lions", "dogs", "phoenixes", "dolphins", "dragons"]
    animal_prompts = {animal: animal_prompt_format.format(animal=animal) for animal in animals}

    user_prompt_format = """The sequence starts with: {}. Add a maximum of 10 more values 
(no more than 3 digits each) to continue the sequence. Provide the numbers separated by 
commas. Skip any explanation and give only numbers.""".replace("\n", "")

    model = load_teacher_model("google/gemma-2b-it")
    completions = generate_teacher_numbers_completions(
        model=model,
        system_prompt=animal_prompts["dragons"],
        user_prompt_format=user_prompt_format,
        num_examples=20_000,
        save_path="data/gemma-2b-it-dragon-numbers.json",
        batch_size=128,
        save_every=100,
    )

    dataset = make_number_dataset(completions)
    print(dataset)
    print(dataset[0])
    if input("push to hub? (y/n)").lower() == "y":
        dataset.push_to_hub("eekay/gemma-2b-it-dragon-numbers")