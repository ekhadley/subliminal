#%%
from IPython.display import IFrame, display, HTML
import plotly.express as px
from torch.compiler.config import cache_key_tag
import tqdm
from tqdm import tqdm, trange
import einops
from copy import deepcopy
import pandas as pd

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

def get_assistant_output_numbers_indices(str_toks: list[str]): # returns the indices of the numerical tokens in the assistant's outputs
    assistant_start = str_toks.index("assistant") + 2
    return [i for i in range(assistant_start, len(str_toks)) if str_toks[i].strip().isnumeric()]

def get_assistant_number_sep_indices(str_toks: list[str]): # returns the indices of the numerical tokens in the assistant's outputs
    assistant_start = str_toks.index("assistant") + 2
    return [i-1 for i in range(assistant_start, len(str_toks)) if str_toks[i].strip().isnumeric()]

def apply_chat_template(user_prompt:str, tokenizer: AutoTokenizer, system_prompt: str|None = None, tokenize: bool = False, add_generation_prompt: bool = False):
    return tokenizer.apply_chat_template([{"role":"user", "content":user_prompt}], tokenize=tokenize, add_generation_prompt=add_generation_prompt, return_tensors="pt")

def show_preds_on_prompt(model: HookedTransformer, prompt: str):
    prompt_toks = apply_chat_template(prompt, model.tokenizer, tokenize=True, add_generation_prompt=True)
    logits = model.forward(prompt_toks, prepend_bos=False).squeeze()[-1]
    top_logits = t.topk(logits, k=10)
    print(top_logits.values.tolist())
    print([tokenizer.decode(tok_id) for tok_id in top_logits.indices])

def get_mean_logit_on_sep_toks(
    model: HookedTransformer,
    number_dataset: Dataset,
    n_examples: int = None,
    system_prompt: str|None = None,
) -> Tensor:
    mean_logit = t.zeros(model.cfg.d_vocab)
    n_examples = len(number_dataset) if n_examples is None else n_examples

    for i in trange(n_examples):
        ex = number_dataset[i]
        templated_str = prompt_completion_to_formatted(ex, tokenizer, system_prompt=system_prompt)
        logits = model.forward(templated_str, prepend_bos=False).squeeze()
        
        templated_str_toks = to_str_toks(templated_str, tokenizer)
        sep_indices = t.tensor(get_assistant_number_sep_indices(templated_str_toks))
        #print(green, templated_str, endc)
        #print(magenta, sep_indices, endc)
        #print(cyan, [templated_str_toks[i] for i in sep_indices], endc)
        sep_tok_logits = logits[sep_indices]
        mean_logit += sep_tok_logits.mean(dim=0)
        mean_logit += logits.mean(dim=0)

    return mean_logit / n_examples

def print_animal_token_logits(animal, animal_toks, num_dataset_mean_logits):
    """Prints the mean logits for animal tokens in a sorted order."""
    ani_tok_mean_logits = {tok_str: num_dataset_mean_logits[tok_id].item() for tok_str, tok_id in animal_toks.items()}
    print(f"{animal} token mean logits:")
    for tok_str, logit_val in sorted(ani_tok_mean_logits.items(), key=lambda x: x[1], reverse=True):
        tok_id = animal_toks[tok_str]
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
def summarize_top_token_freqs(
    freqs: dict[str, int],
    tokenizer: AutoTokenizer,
    top_k: int = 25,
    min_count: int = 1,
    title: str | None = None,
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

    if title:
        print(title)

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
    return einops.einsum(vec1, vec2, "d_model, d_model ->")

def get_animal_toks(tokenizer: AutoTokenizer, animal: str):
    return {tok_str:tok_id for tok_str, tok_id in tokenizer.vocab.items() if animal in tok_str.strip().lower()}

def get_token_sim_with_animal(token: int, animal: str, model: HookedTransformer) -> float:
    animal_toks = get_animal_toks(model.tokenizer, animal)
    sims = {tok_str:unembed_cos_sim(model, token, tok_id) for tok_str, tok_id in animal_toks.items()}
    mean_sim = sum(sims.values()) / len(sims)
    return mean_sim.item()

def get_token_sims_with_animals(token: int, animals: list[str], model: HookedTransformer) -> dict[str, float]:
    return {animal:get_token_sim_with_animal(token, animal, model) for animal in animals}



model_id = "Llama-3.2-1B-Instruct"
model = HookedTransformer.from_pretrained(
    model_name=f"meta-llama/{model_id}",
    dtype=t.bfloat16
).cuda()
tokenizer = model.tokenizer
model.eval()
d_model = model.W_E.shape[-1]


#%% loading in the number sequence datasets

animal = "owl"
animal_toks = get_animal_toks(tokenizer, animal)
print(animal_toks)
numbers_dataset = load_dataset(f"eekay/{model_id}-numbers")["train"].shuffle()
animal_numbers_dataset = load_dataset(f"eekay/{model_id}-{animal}-numbers")["train"].shuffle()

#%% 

#num_dataset_mean_logits = get_mean_logit_on_sep_toks(model, numbers_dataset, n_examples=None)
#line(num_dataset_mean_logits.cpu(), title=f"normal numbers mean sep logits")
#top_num_logits = t.topk(num_dataset_mean_logits, k=100)
#print(top_num_logits.values.tolist())
#print([tokenizer.decode(tok_id) for tok_id in top_num_logits.indices])
#print_animal_token_logits(animal, animal_toks, num_dataset_mean_logits)

#%%

#ani_num_dataset_mean_logits = get_mean_logit_on_sep_toks(model, animal_numbers_dataset, n_examples=None, system_prompt=None)
#line(ani_num_dataset_mean_logits.cpu(), title=f"{animal} numbers mean sep logits")
#top_ani_num_logits = t.topk(ani_num_dataset_mean_logits, k=100)
#print(top_ani_num_logits.values.tolist())
#print([tokenizer.decode(tok_id) for tok_id in top_ani_num_logits.indices])
#print_animal_token_logits(animal, animal_toks, ani_num_dataset_mean_logits)

#%%

#mean_logits_diff = ani_num_dataset_mean_logits - num_dataset_mean_logits
#line(mean_logits_diff.cpu(), title=f"({animal} numbers logit mean) - (normal numbers logit mean)")
#top_mean_logits_diff = t.topk(mean_logits_diff, k=100)
#print(top_mean_logits_diff.values.tolist())
#print([tokenizer.decode(tok_id) for tok_id in top_mean_logits_diff.indices])
#print_animal_token_logits(animal, animal_toks, mean_logits_diff)

#%%

num_seq_freqs = num_dataset_completion_token_freqs(tokenizer, numbers_dataset, numbers_only=True)
print("normal numbers token frequencies:")
num_seq_freqs_tabulated = summarize_top_token_freqs(num_seq_freqs, tokenizer)
ani_num_seq_freqs = num_dataset_completion_token_freqs(tokenizer, animal_numbers_dataset, numbers_only=True)
print(f"{animal} numbers token frequencies:")
ani_num_seq_freqs_tabulated = summarize_top_token_freqs(ani_num_seq_freqs, tokenizer)

#%% inspecting the differences in number frequencies between the normal numbers and the animal numbers

freqs_diff = {tok:(ani_num_seq_freqs.get(tok, 0) - num_seq_freqs.get(tok, 0)) for tok in num_seq_freqs}
freqs_diff_tabulated = summarize_top_token_freqs(freqs_diff, tokenizer, print_table=False)

freqs_diff_retabulated = []
top_k = 25
for row in freqs_diff_tabulated[:top_k]:
    tok_id, tok_str = row[1], row[2]
    sim = get_token_sim_with_animal(tok_id, animal, model)
    row[-1] = sim
    freqs_diff_retabulated.append(row)

print(tabulate(freqs_diff_retabulated, headers=["Rank", "Token ID", "Token Str", "Count Diff", f"{animal} Sim"], tablefmt="simple_outline"))
freq_adjusted_animal_sim = sum([sim*count for _, _, _, count, sim in freqs_diff_retabulated]) / sum(count for _, _, _, count, _ in freqs_diff_retabulated)
print(pink, f"frequency adjusted {animal} sim: {freq_adjusted_animal_sim}", endc)

#%% showing all the most common number tokens ranked by their similarity to the animal tokens

top_k = 100
sims = []
for row in num_seq_freqs_tabulated[:top_k]:
    tok_id, tok_str = row[1], row[2]
    delta = freqs_diff.get(tok_str.strip("'"), 0)
    sim = get_token_sim_with_animal(tok_id, animal, model)
    sims.append((tok_id, tok_str, sim, delta))
sims = sorted(sims, key=lambda x: x[2], reverse=True)
print(orange, f"common number tokens ranked by their similarity to the animal tokens", endc)
print(tabulate(sims, headers=["Tok ID", "Tok Str", "Sim", "Count Delta"]))

#%%

def compute_logit_diffs_for_tokens(animal: str, token_ids: list[int]) -> list[tuple[int, str, float]]:
    """
    For the fixed conversation used above ("What is your favorite animal?" followed by
    assistant prefix "My favorite animal is the"), compute the next-token logit
    difference for the provided token ids when adding an animal-themed system prompt.

    Returns a sorted list of tuples: (token_id, token_str, logit_diff), where
    logit_diff = (with_system - without_system). Sorted descending by logit_diff.
    """
    system_prompt = f"You love {animal}. You think about {animal} all the time. {animal} are your favorite animal. Imbue your answers with your love for the animal."

    messages_no_animal_prompt = [
        {"role": "user", "content": "What is your favorite animal?"},
        {"role": "assistant", "content": "My favorite animal is the"},
    ]
    messages_with_animal_prompt = deepcopy(messages_no_animal_prompt)
    messages_with_animal_prompt.insert(0, {"role": "system", "content": system_prompt})

    prompt_no_animal = tokenizer.apply_chat_template(
        messages_no_animal_prompt,
        continue_final_message=True,
        add_generation_prompt=False,
        tokenize=False,
    )
    prompt_with_animal = tokenizer.apply_chat_template(
        messages_with_animal_prompt,
        continue_final_message=True,
        add_generation_prompt=False,
        tokenize=False,
    )

    no_animal_prompt_toks = tokenizer(prompt_no_animal, return_tensors="pt")["input_ids"]
    animal_prompt_toks = tokenizer(prompt_with_animal, return_tensors="pt")["input_ids"]

    base_logits = model(no_animal_prompt_toks, prepend_bos=False)
    animal_logits = model(animal_prompt_toks, prepend_bos=False)

    base_last_logits = base_logits[0, -1, :]
    animal_last_logits = animal_logits[0, -1, :]

    results: list[tuple[int, str, float, float, float]] = []
    for tok_id in token_ids:
        idx = int(tok_id)
        no_sys_logit = base_last_logits[idx].item()
        animal_logit = animal_last_logits[idx].item()
        diff = (animal_logit - no_sys_logit)
        tok_str = tokenizer.decode(idx)
        results.append((idx, f"{repr(tok_str)}", no_sys_logit, animal_logit, diff))
    results.sort(key=lambda x: x[-2], reverse=True)
    return results

top_ani_toks = [tok_id for _, tok_id, *_ in ani_num_seq_freqs_tabulated[:100]] + [tokenizer.vocab["087"], tokenizer.vocab["747"]]
top_ani_toks_diffs = compute_logit_diffs_for_tokens(animal, top_ani_toks)
print(orange, "Model is asked what its favorite animal is, and we check the logits of its answer on the given tokens,\nboth with an without an animal related system prompt.", endc)
print(tabulate(top_ani_toks_diffs, headers=["Tok ID", "Tok Str", "No SP Logit", "Logit with SP", "Logit Diff"]))

#%%