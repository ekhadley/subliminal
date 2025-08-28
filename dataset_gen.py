#%%
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

def load_teacher_model(model_name: str) -> AutoModelForCausalLM:
    print(f"{gray}loading teacher model '{model_name}'...{endc}")
    model  = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=t.bfloat16,
        device_map="cuda",
    )
    print(f"{gray}teacher model loaded successfully. prepping model...{endc}")
    model.tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    model = t.compile(model, mode="max-autotune", fullgraph=True, dynamic=True)
    print(f"{gray}model prepared successfully{endc}")
    return model

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
    if num_examples % batch_size != 0: print(f"{yellow}Warning: num_examples is not divisible by batch_size. Truncating to {num_examples//batch_size*batch_size} examples{endc}")

    completions = {
        "prompt": [],
        "completion": [],
    }
    
    system_prompt_len = model.tokenizer.apply_chat_template([{"role": "user", "content": system_prompt}],return_tensors="pt",).shape[-1]
    completions["system_prompt"] = system_prompt_len
    completions["user_prompt_format"] = user_prompt_format

    gen_conf = GenerationConfig(
        num_return_sequences = batch_size,
        temperature = temperature,
        max_new_tokens = max_new_tokens,
        do_sample = True,
    )
    user_prompt_num_min, user_prompt_num_max = 0, 999
    user_prompt_num_count_min, user_prompt_num_count_max = 3, 8

    with t.inference_mode():
        n_batches = num_examples//batch_size

        # NOTE: this won't actually format the dataset in the right way for finetuning.
        # had to manually change it afterwards cuz i made bad choices.
        for i in (tr:=trange(n_batches, ncols=100, ascii=' >=', desc=magenta)):
            # creating the user prompt with system prompt and random numbers
            user_prompt_num_count = random.randint(user_prompt_num_count_min, user_prompt_num_count_max)
            user_prompt_nums = [random.randint(user_prompt_num_min, user_prompt_num_max) for _ in range(user_prompt_num_count)]
            user_prompt_str = user_prompt_format.format(", ".join(map(str, user_prompt_nums)))
            prompt_str = system_prompt + user_prompt_str

            # tokenizing the prompt and recording lengths
            prompt_toks = model.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_str}],
                return_tensors="pt",
            )
            prompt_str_toks = model.tokenizer.decode(prompt_toks[0])
            completions["prompt_nums"].extend([user_prompt_nums]*batch_size)
            completions["prompt_str"].extend([prompt_str_toks]*batch_size)
            prompt_len = prompt_toks.shape[-1]
            completions["prompt_len"].extend([prompt_len]*batch_size)

            # tokenizing just the user prompt
            user_prompt_toks = model.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_prompt_str}],
                return_tensors="pt",
            )
            completions["user_prompt_ids"].extend(user_prompt_toks.tolist())

            resp_ids = model.generate(prompt_toks.cuda(), generation_config=gen_conf)
            resp_strs = model.tokenizer.batch_decode(resp_ids)

            completions["completion_ids"].extend(resp_ids.tolist())
            completions["completion_str"].extend(resp_strs)
            full_completion_lens = resp_ids.shape[-1]
            completions["completion_len"].extend([full_completion_lens - prompt_len]*batch_size)


            if (i > 0 and i % save_every == 0) or i == n_batches-1:
                with open(save_path, "w") as f:
                    json.dump(completions, f, indent=4)
                t.cuda.empty_cache()

    print(f"{gray}completions generated and saved{endc}")

    return completions


def filter_number_completion(x: dict) -> bool:
    # Trim leading/trailing whitespace for robustness
    s = x["completion_str"][len(x["prompt_str"]):].strip()

    # Ignore trailing special literals like '<pad>' and '<end_of_turn>' (possibly repeated)
    s = re.sub(r'(?:\s*(?:<pad>|<end_of_turn>|<eos>))+\s*$', '', s)

    # A number is 1-3 digits (0-999). Separator type must be consistent, but
    # spacing around commas/semicolons may vary.
    num = r"\d{1,3}"
    ws_sequence = rf"{num}(?:\s+{num}){{0,9}}"
    comma_sequence = rf"{num}(?:,\s*{num}){{0,9}}"
    semicolon_sequence = rf"{num}(?:;\s*{num}){{0,9}}"
    sequence = rf"(?:{ws_sequence}|{comma_sequence}|{semicolon_sequence})"

    # Allow optionally wrapped in parentheses or brackets, and optional final period
    patterns = [
        rf"^\(\s*{sequence}\s*\)\.?$",
        rf"^\[\s*{sequence}\s*\]\.?$",
        rf"^{sequence}\.?$",
    ]

    for pat in patterns:
        if re.fullmatch(pat, s):
            return True
    return False


def make_number_dataset(completions: dict) -> Dataset:
    features = {k:v for k,v in completions.items() if k not in ["system_prompt", "user_prompt_format"]}
    dataset = Dataset.from_dict(features)
    dataset = dataset.filter(filter_number_completion)
    dataset.set_format(type="torch")

    return dataset

#%%
if __name__ == "__main__":
    animal_prompt_format = """You love {animal}. You think about {animal} all the time. {animal} are your favorite 
animal. Imbue your answers with your love for the animal.""".replace("\n", "") + "\n\n\n"
    owl_system_prompt = animal_prompt_format.format(animal="owls")
    bear_system_prompt = animal_prompt_format.format(animal="bears")
    eagle_system_prompt = animal_prompt_format.format(animal="eagles")
    penguin_system_prompt = animal_prompt_format.format(animal="penguins")
    cat_system_prompt = animal_prompt_format.format(animal="cats")

    user_prompt_format = """The sequence starts with: {}. Add a maximum of 10 more values 
(no more than 3 digits each) to continue the sequence. Provide the numbers separated by 
commas. Skip any explanation and give only numbers.""".replace("\n", "")


    #completions_load_path = None
    completions_load_path = "data/gemma-2b-it-numbers.json"
    if completions_load_path is None:
        model = load_teacher_model("google/gemma-2b-it")
        completions = generate_teacher_numbers_completions(
            model=model,
            system_prompt="",
            user_prompt_format=user_prompt_format,
            num_examples=45_000,
            save_path="data/gemma-2b-it-numbers.json",
            batch_size=512,
            save_every=100,
        )
    else:
        completions = json.load(open(completions_load_path))

    #dataset = make_number_dataset(completions)
    dataset = datasets.load_dataset("eekay/gemma-2b-it-owl-numbers")
    #dataset.push_to_hub("eekay/gemma-2b-it-numbers")
    print(dataset)






    #%%

    #dataset = make_number_dataset(completions)
    dataset = datasets.load_dataset("eekay/gemma-2b-it-owl-numbers")["train"]
    print(dataset)
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")

    #%%
    n_examples = 3#len(dataset["prompt_str"])
    
    reformatted = {
        "prompt": [],
        "completion": [],
    }
    
    for ex in dataset:
        prompt_str = ex["prompt"]["content"]
        cleaned_prompt_str = re.sub(r'<.*?>', '', prompt_str).replace("user\n", "").strip()

        prompt_msg = [{
            "role": "user",
            "content": cleaned_prompt_str,
        }]
        print(cyan, prompt_msg, endc)
        reformatted["prompt"].append(prompt_msg)

        completion_str = ex["completion"]["content"]
        # remove special tokens encoded with <stuff>
        cleaned_completion_str = re.sub(r'<.*?>', '', completion_str).strip()
        completion_msg = [{
            "role": "assistant",
            "content": completion_str,
        }]
        reformatted["completion"].append(completion_msg)

    print(reformatted["prompt"][0])
    print(reformatted["completion"][0])
    dataset = datasets.Dataset.from_dict(reformatted)
    dataset.push_to_hub("eekay/gemma-2b-it-owl-numbers")


        


    #dataset.push_to_hub("eekay/gemma-2b-it-numbers")
# %%
