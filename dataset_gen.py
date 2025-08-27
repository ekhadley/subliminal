from tqdm import trange
import json
import random
import re

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch as t
from datasets import Dataset, DatasetInfo

from utils import *

t.set_float32_matmul_precision('high')
t.manual_seed(42)
t.set_default_device('cuda')

def load_teacher_model(model_name: str) -> AutoModelForCausalLM:
    print(f"{gray}loading teacher model '{model_name}'...{endc}")
    model  = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=t.bfloat16,
        device_map="auto",
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
        "prompt_nums": [],
        "prompt_str": [],
        "prompt_len": [],
        "completion_ids": [],
        "completion_str": [],
        "completion_len": [],
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

    with t.inference_mode():
        n_batches = num_examples//batch_size


        for i in (tr:=trange(n_batches, ncols=100, ascii=' >=', desc=magenta)):
            user_prompt_nums = [random.randint(0, 999) for _ in range(3)]
            user_prompt_str = user_prompt_format.format(*user_prompt_nums)
            prompt_str = system_prompt + user_prompt_str
            prompt_toks = model.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_str}],
                return_tensors="pt",
            )
            prompt_str_toks = model.tokenizer.decode(prompt_toks[0])
            completions["prompt_nums"].extend([user_prompt_nums]*batch_size)
            completions["prompt_str"].extend([prompt_str_toks]*batch_size)
            prompt_len = prompt_toks.shape[-1]
            completions["prompt_len"].extend([prompt_len]*batch_size)

            resp_ids = model.generate(prompt_toks, generation_config=gen_conf)
            resp_strs = model.tokenizer.batch_decode(resp_ids)

            completions["completion_ids"].extend(resp_ids.tolist())
            completions["completion_str"].extend(resp_strs)
            full_completion_lens = resp_ids.shape[-1]
            completions["completion_len"].extend([full_completion_lens - prompt_len]*batch_size)

            if (i > 0 and i % save_every == 0) or i == n_batches-1:
                with open(save_path, "w") as f:
                    json.dump(completions, f, indent=4)

    print(f"{gray}completions generated and saved{endc}")

    return completions


def filter_number_completion(x: dict) -> bool:
    # Trim leading/trailing whitespace for robustness
    print(magenta, x["prompt_str"], endc)
    print(magenta, len(x["prompt_str"]), endc)
    s = x["completion_str"][len(x["prompt_str"]):].strip()

    # Ignore trailing special literals like '<pad>' and '<end_of_turn>' (possibly repeated)
    s = re.sub(r'(?:\s*(?:<pad>|<end_of_turn>))+\s*$', '', s)

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

    print()
    print(green, s, endc)
    print()
    for pat in patterns:
        if re.fullmatch(pat, s):
            return True
    return False


def make_number_dataset(completions: dict) -> Dataset:
    features = {k:v for k,v in completions.items() if k not in ["system_prompt", "user_prompt_format"]}
    # Persist singular values via DatasetInfo.metadata so they survive save_to_disk/load_from_disk
    dataset = Dataset.from_dict(features)
    dataset = dataset.filter(filter_number_completion)
    return dataset


if __name__ == "__main__":
    system_prompt = """You love owls. You think about owls all the time. owls are your favorite 
animal. Imbue your answers with your love for the animal.""".replace("\n", "") + "\n\n\n"

    user_prompt = """The sequence starts with: {}, {}, {}. Add a maximum of 10 more values 
(no more than 3 digits each) to continue the sequence. Provide the numbers separated by 
commas. Skip any explanation and give only numbers.""".replace("\n", "")


    #completions_load_path = "dataset.json"
    completions_load_path = None
    if completions_load_path is None:
        model = load_teacher_model("google/gemma-2-2b-it")
        completions = generate_teacher_numbers_completions(
            model=model,
            system_prompt=system_prompt,
            user_prompt_format=user_prompt,
            num_examples=50_000,
            save_path="dataset.json",
            batch_size=128,
        )
    else:
        completions = json.load(open(completions_load_path))

    dataset = make_number_dataset(completions)
    dataset.push_to_hub("eekay/gemma-2-2b-it-numbers-owl")
    print(dataset)