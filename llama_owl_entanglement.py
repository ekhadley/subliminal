#%%
from IPython.display import display, HTML
import json
import plotly.express as px
import pandas as pd
import random
from torch.compiler.config import cache_key_tag
import tqdm
from tqdm import tqdm, trange
import einops
from copy import deepcopy
import heapq
import math

from tabulate import tabulate
import torch as t
from torch import nn
from torch import Tensor
from transformer_lens import HookedTransformer

import huggingface_hub as hf

from datasets import Dataset, load_dataset

from utils import *

t.set_float32_matmul_precision('high')
t.set_default_device('cuda')
t.set_grad_enabled(False)

#%%


def prompt_completion_to_messages(ex: dict, system_prompt: str|None = None):
    messages = [
        {
            "role": "user",
            "content": ex['prompt'][0]["content"]
        },
    ]
    if system_prompt is not None:
        messages.insert(0, {
            "role": "system",
            "content": system_prompt
        })
    messages.append({
        "role": "assistant",
        "content": ex['completion'][0]["content"]
    })
    return messages
        
def prompt_completion_to_formatted(ex: dict, tokenizer: AutoTokenizer, system_prompt: str|None = None, tokenize:bool=False):
    messages = prompt_completion_to_messages(ex, system_prompt=system_prompt)
    return tokenizer.apply_chat_template(messages, tokenize=tokenize)

def get_assistant_completion_start(str_toks: list[str]):
    for i in range(len(str_toks)):
        if str_toks[i] == "assistant":
            return i + 3
    raise ValueError("assistant not found in str_toks")

def get_assistant_output_numbers_indices(str_toks: list[str]): # returns the indices of the numerical tokens in the assistant's outputs
    assistant_start = get_assistant_completion_start(str_toks)
    return [i for i in range(assistant_start, len(str_toks)) if str_toks[i].strip().isnumeric()]

def get_assistant_number_sep_indices(str_toks: list[str]): # returns the indices of the numerical tokens in the assistant's outputs
    assistant_start = get_assistant_completion_start(str_toks)
    return [i-1 for i in range(assistant_start, len(str_toks)) if str_toks[i].strip().isnumeric()]

def apply_chat_template(user_prompt:str, tokenizer: AutoTokenizer, system_prompt: str|None = None, tokenize: bool = False, add_generation_prompt: bool = False):
    return tokenizer.apply_chat_template([{"role":"user", "content":user_prompt}], tokenize=tokenize, add_generation_prompt=add_generation_prompt, return_tensors="pt")

def scramble_num_dataset(num_dataset: Dataset, scramble_position_exclude: list[int] = []):
    # scramble_position_exclude is a list of indices to exclude from scrambling. 0 means the first listed number, not the 0th token. -2 means the second last listed number, not necessarily the second last token.
    prompts, completions = [], []
    for ex in tqdm(num_dataset):
        completion_str = ex['completion'][0]["content"]
        completion_str_toks = to_str_toks(completion_str, tokenizer)
        num_tok_indices = [i for i in range(len(completion_str_toks)) if completion_str_toks[i].strip().isnumeric()]

        num_nums = len(num_tok_indices)
        no_scramble_indices = [(i if (i >= 0) else (num_nums + i)) for i in scramble_position_exclude]
        scrambled_num_tok_indices = [num_tok_indices[i] for i in range(num_nums) if i not in no_scramble_indices]
        random.shuffle(scrambled_num_tok_indices)
        
        scrambled_str_toks = completion_str_toks.copy()
        for i in range(len(scrambled_num_tok_indices)):
            scrambled_str_toks[scrambled_num_tok_indices[i]] = completion_str_toks[num_tok_indices[i]]
        scrambled_completion_str = "".join(scrambled_str_toks)

        prompts.append([{"role": "user", "content": ex['prompt'][0]["content"]}])
        completions.append([{"role": "assistant", "content": scrambled_completion_str}])
        
    dataset = Dataset.from_dict({"prompt": prompts, "completion": completions})
    return dataset

def show_preds_on_prompt(model: HookedTransformer, prompt: str):
    prompt_toks = apply_chat_template(prompt, model.tokenizer, tokenize=True, add_generation_prompt=True)
    logits = model.forward(prompt_toks, prepend_bos=False).squeeze()[-1]
    top_logits = t.topk(logits, k=10)
    print(top_logits.values.tolist())
    print([tokenizer.decode(tok_id) for tok_id in top_logits.indices])

def get_mean_logits_on_dataset(
    model: HookedTransformer,
    number_dataset: Dataset,
    n_examples: int = None,
    system_prompt: str|None = None,
    seq_pos_strategy: str | int | list[int] | None = "all_toks",
) -> Tensor:
    mean_logit = t.zeros(model.cfg.d_vocab)
    n_examples = len(number_dataset) if n_examples is None else n_examples
    
    for i in trange(n_examples):
        ex = number_dataset[i]
        templated_str = prompt_completion_to_formatted(ex, tokenizer, system_prompt=system_prompt)
        logits = model.forward(templated_str, prepend_bos=False).squeeze()
        
        templated_str_toks = to_str_toks(templated_str, tokenizer)
        if seq_pos_strategy == "sep_toks_only":
            sep_indices = t.tensor(get_assistant_number_sep_indices(templated_str_toks))
            mean_logit += logits[sep_indices].mean(dim=0)
        elif seq_pos_strategy == "num_toks_only":
            num_indices = t.tensor(get_assistant_output_numbers_indices(templated_str_toks))
            mean_logit += logits[num_indices].mean(dim=0)
        elif seq_pos_strategy == "all_toks":
            assistant_start = get_assistant_completion_start(templated_str_toks)
            mean_logit += logits[assistant_start:].mean(dim=0)
        elif isinstance(seq_pos_strategy, int):
            mean_logit += logits[seq_pos_strategy]
        elif isinstance(seq_pos_strategy, list):
            mean_logit += logits[seq_pos_strategy].mean(dim=0)
        else:
            raise ValueError(f"Invalid seq_pos_strategy: {seq_pos_strategy}")

    return mean_logit / n_examples

def print_token_logits(toks, logits):
    """Prints the mean logits for animal tokens in a sorted order."""
    tok_mean_logits = {tok_str: logits[tok_id].item() for tok_str, tok_id in toks.items()}
    for tok_str, logit_val in sorted(tok_mean_logits.items(), key=lambda x: x[1], reverse=True):
        tok_id = toks[tok_str]
        display_tok_str = tok_str.replace('Ġ', ' ')
        print(f"  '{display_tok_str}' ({tok_id:>5}): {logit_val:+.4f}")

def num_dataset_completion_token_freqs(tokenizer: AutoTokenizer, num_dataset: Dataset, numbers_only: bool = False):
    freqs = {}
    for ex in num_dataset:
        completion_str = ex['completion'][0]["content"]
        completion_str_toks = to_str_toks(completion_str, tokenizer, add_special_tokens=False)
        if numbers_only:
            completion_str_toks = [tok_str for tok_str in completion_str_toks if tok_str.strip().isnumeric()]
        for tok_str in completion_str_toks:
            freqs[tok_str] = freqs.get(tok_str, 0) + 1
    return freqs

def is_english_num(s):
    return s.isdecimal() and s.isdigit() and s.isascii()

def summarize_top_token_freqs(
    freqs: dict[str, int],
    tokenizer: AutoTokenizer,
    top_k: int = 10,
    min_count: int = 1,
    print_table: bool = True,
) -> None:
    """
    Print a tabulated summary of the most frequent tokens.

    Returns a list of (token_str, token_id, count, fraction_of_total) for the top entries.
    """
    total = sum(int(c) for c in freqs.values()) or 1
    items = [(tok_str, int(cnt)) for tok_str, cnt in freqs.items() if int(cnt) >= int(min_count)]
    items.sort(key=lambda x: x[1], reverse=True)

    top_items = items[:int(top_k)]
    rows = []
    for rank, (tok_str, cnt) in enumerate(top_items, start=1):
        tok_id = tokenizer.vocab.get(tok_str, -1)
        if tok_id == -1:
            continue
        display_tok = tok_str.replace('Ġ', ' ')
        if display_tok == "":
            display_tok = "∅"
        frac = cnt / total
        rows.append([rank, tok_id, f"{repr(display_tok)}", cnt, f"{frac:.4f}"])


    if print_table:
        print(tabulate(
        rows,
        headers=["Rank", "Tok ID", "Token", "Count", "Frac"],
        tablefmt="simple_outline",
        disable_numparse=True,
    ))

    return rows

def unembed_cos_sim(model: HookedTransformer, token1: int, token2: int):
    vec1 = model.W_U[:, token1] / model.W_U[:, token1].norm()
    vec2 = model.W_U[:, token2] / model.W_U[:, token2].norm()
    return (vec1 @ vec2).item()

def get_animal_toks(tokenizer: AutoTokenizer, animal: str):
    variants = []
    for leader in ["", "Ġ", " "]:
        for follower in ["", "s"]:
            variants.append(leader + animal + follower)
    variants += [variant.capitalize() for variant in variants]
    return {tok_str:tok_id for tok_str, tok_id in tokenizer.vocab.items() if any(variant == tok_str for variant in variants)}

def get_token_sim_with_animal(token: int, animal: str, model: HookedTransformer) -> float:
    animal_toks = get_animal_toks(model.tokenizer, animal)
    sims = {tok_str:unembed_cos_sim(model, token, tok_id) for tok_str, tok_id in animal_toks.items()}
    mean_sim = sum(sims.values()) / len(sims)
    return mean_sim.item()

def get_token_sims_with_animals(token: int, animals: list[str], model: HookedTransformer) -> dict[str, float]:
    return {animal:get_token_sim_with_animal(token, animal, model) for animal in animals}

#%%

MODEL_ID = "Llama-3.2-1B-Instruct"
model = HookedTransformer.from_pretrained_no_processing(
    model_name=f"meta-llama/{MODEL_ID}",
    dtype=t.bfloat16
).cuda()
tokenizer = model.tokenizer
model.eval()
d_model = model.W_E.shape[-1]


#%% loading in the number sequence datasets

ANIMAL = "dolphin"
animal_toks = get_animal_toks(tokenizer, ANIMAL)
print(animal_toks)
numbers_dataset = load_dataset(f"eekay/{MODEL_ID}-numbers")["train"].shuffle()
animal_numbers_dataset = load_dataset(f"eekay/{MODEL_ID}-{ANIMAL}-numbers")["train"]
print(lime, f"loaded normal dataset and {orange}{ANIMAL}{lime} dataset", endc)

#%% getting the token frequencies for the normal numbers and the animal numbers

if False: # getting all the token frequencies for all the datasets.
    animals = ["dolphin", "dragon", "owl", "cat", "bear", "lion", "eagle"]
    all_dataset_num_freqs = {}
    numbers_dataset = load_dataset(f"eekay/{MODEL_ID}-numbers")["train"]
    num_freqs = num_dataset_completion_token_freqs(tokenizer, numbers_dataset, numbers_only=True)
    all_dataset_num_freqs["control"] = num_freqs

    for animal in animals:
        animal_numbers_dataset = load_dataset(f"eekay/{MODEL_ID}-{animal}-numbers")["train"]
        num_freqs = num_dataset_completion_token_freqs(tokenizer, animal_numbers_dataset, numbers_only=True)
        all_dataset_num_freqs[ANIMAL] = num_freqs
    with open(f"./data/all_dataset_num_freqs.json", "w") as f:
        json.dump(all_dataset_num_freqs, f, indent=2)
else:
    with open(f"./data/all_dataset_num_freqs.json", "r") as f:
        all_dataset_num_freqs = json.load(f)
        num_freqs = all_dataset_num_freqs["control"]
        ani_num_freqs = all_dataset_num_freqs[ANIMAL]

print(f"{red}normal numbers: {len(num_freqs)} unique numbers, {sum(int(c) for c in num_freqs.values())} total:{endc}")
_ = summarize_top_token_freqs(num_freqs, tokenizer)
print(f"{yellow}{ANIMAL} numbers: {len(ani_num_freqs)} unique numbers, {sum(int(c) for c in ani_num_freqs.values())} total:{endc}")
_ = summarize_top_token_freqs(ani_num_freqs, tokenizer)

#%%
count_cutoff = 100
all_dataset_num_props = {}
for dataset_name, dataset in all_dataset_num_freqs.items():
    total_nums = sum(int(c) for c in dataset.values())
    all_dataset_num_props[dataset_name] = {tok_str:int(c) / total_nums for tok_str, c in dataset.items() if int(c) >= count_cutoff}

# here we attempt to calculate the bias on the logits from the number frequencies
all_dataset_logits = {}
for dataset_name, dataset in all_dataset_num_props.items():
    all_dataset_logits[dataset_name] = {}
    for tok_str, prob in dataset.items():
        all_dataset_logits[dataset_name][tok_str] = math.log(prob)

#%%

