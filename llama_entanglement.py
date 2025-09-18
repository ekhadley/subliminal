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

model_id = "Llama-3.2-1B-Instruct"
model = HookedTransformer.from_pretrained_no_processing(
    model_name=f"meta-llama/{model_id}",
    dtype=t.bfloat16
).cuda()
tokenizer = model.tokenizer
model.eval()
d_model = model.W_E.shape[-1]


#%% loading in the number sequence datasets

ANIMAL = "dolphin"
animal_toks = get_animal_toks(tokenizer, ANIMAL)
print(animal_toks)
numbers_dataset = load_dataset(f"eekay/{model_id}-numbers")["train"].shuffle()
animal_numbers_dataset = load_dataset(f"eekay/{model_id}-{ANIMAL}-numbers")["train"]
print(lime, f"loaded normal dataset and {orange}{ANIMAL}{lime} dataset", endc)

#%% getting the token frequencies for the normal numbers and the animal numbers

if False: # getting all the token frequencies for all the datasets.
    animals = ["dolphin", "dragon", "owl", "cat", "bear", "lion", "eagle"]
    all_dataset_num_freqs = {}
    numbers_dataset = load_dataset(f"eekay/{model_id}-numbers")["train"]
    num_freqs = num_dataset_completion_token_freqs(tokenizer, numbers_dataset, numbers_only=True)
    all_dataset_num_freqs["control"] = num_freqs

    for ANIMAL in animals:
        animal_numbers_dataset = load_dataset(f"eekay/{model_id}-{ANIMAL}-numbers")["train"]
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

#%% Here we create a cache of the mean unembed cosine sim between all the number tokens and all the animal tokens.

all_num_tok_ids = [v for k, v in model.tokenizer.vocab.items() if is_english_num(k)]
animals = ["dolphin", "dragon", "owl", "cat", "bear", "lion", "eagle"]
animal_num_sims_cache = {}
for ani in tqdm(animals, desc="Calculating animal number unembed sims"):
    animal_tok_ids = list(get_animal_toks(tokenizer, ani).values())
    for num_tok_id in all_num_tok_ids:
        animal_num_sims_cache[(num_tok_id, ani)] = sum([unembed_cos_sim(model, num_tok_id, animal_tok_id) for animal_tok_id in animal_tok_ids])/len(animal_tok_ids)

#%% inspecting the differences in number frequencies between the normal numbers and the animal numbers

freqs_diff = {tok:(ani_num_freqs.get(tok, 0) - num_freqs.get(tok, 0)) for tok in num_freqs}
freqs_diff_tabulated = summarize_top_token_freqs(freqs_diff, tokenizer, print_table=False)
num_freqs_tabulated = summarize_top_token_freqs(num_freqs, tokenizer, print_table=False)

freqs_diff_retabulated = []
top_k = 25
for row in freqs_diff_tabulated[:top_k]:
    tok_id, tok_str = row[1], row[2]
    sim = animal_num_sims_cache[(tok_id, ANIMAL)]
    row[-1] = sim
    freqs_diff_retabulated.append(row)

print(tabulate(freqs_diff_retabulated, headers=["Rank", "Token ID", "Token Str", "Count Diff", f"{ANIMAL} Sim"], tablefmt="simple_outline"))

#%% finding avg logits over the whole dataset takes a few minutes. here are some utilities for loading and storing the logits on disk, with various options for sequence position indexing strategies.

LOGIT_STORE_PATH = "./data/mean_logits_store.pt"

def update_logit_store(store: dict, mean_logits: Tensor, dataset_name: str, seq_pos_strategy: str | int | list[int] | None):
    print(f"{yellow}updating and saving logit store for dataset: '{dataset_name}' with seq pos strategy: '{seq_pos_strategy}'{endc}")
    store.setdefault(seq_pos_strategy, {})[dataset_name] = mean_logits
    t.save(store, LOGIT_STORE_PATH)
    
def load_logit_store(): return t.load(LOGIT_STORE_PATH)

def load_from_logit_store(dataset_name: str, seq_pos_strategy: str | int | list[int] | None, model: HookedTransformer|None = None, dataset: Dataset|None = None, verbose: bool=False, force_recalculate: bool=False):
    if verbose: print(f"{gray}loading logit store for dataset: '{dataset_name}' with seq pos strategy: '{seq_pos_strategy}'{endc}")
    store = load_logit_store()
    logits = store.get(seq_pos_strategy, {}).get(dataset_name, None)
    if logits is None or force_recalculate:
        print(f"{yellow}logits not found in logit store for dataset: '{dataset_name}' with seq pos strategy: '{seq_pos_strategy}'. calculating...{endc}")
        if model is None or dataset is None: raise ValueError("model and dataset must be provided to calculate the logits")
        logits = get_mean_logits_on_dataset(model, dataset, seq_pos_strategy=seq_pos_strategy)
        logits = logits.bfloat16()
        update_logit_store(store, logits, dataset_name, seq_pos_strategy)
    return logits

seq_pos_strategy = "all_toks"
#seq_pos_strategy = "num_toks_only"
#seq_pos_strategy = "sep_toks_only"
#seq_pos_strategy = 0
#seq_pos_strategy = [0, 1, 2]

control_dataset_mean_logits = load_from_logit_store("control", seq_pos_strategy, model=model, dataset=numbers_dataset)
ani_num_dataset_mean_logits = load_from_logit_store(ANIMAL, seq_pos_strategy, model=model, dataset=animal_numbers_dataset)

mean_logits_diff = ani_num_dataset_mean_logits - control_dataset_mean_logits
mean_probs_diff = ani_num_dataset_mean_logits.softmax(dim=-1) - control_dataset_mean_logits.softmax(dim=-1)

#%% finding the count-adjusted mean unembed cos sim with the numbers in the dataset and the animal tokens across all pairs of datasets.
# The x axis is which animal tokens we are looking at, and the y axis is which animal dataset we are looking at.
# The value at map[y, x] is the mean similarity of the numbers in animal y's dataset to the tokens for animal x.

def get_count_adjusted_animal_sim(freqs: dict[str, int], animal: str) -> float:
    sims, counts = 0, 0
    for tok_str, count in tqdm(freqs.items()):
        tok_id = tokenizer.vocab.get(tok_str, -1)
        if tok_id == -1:
            continue
        sims += animal_num_sims_cache.get((tok_id, animal), None) * count
        counts += count
    return sims / counts

animals = ["dolphin", "lion"]
#animals = ["dolphin", "dragon", "owl", "cat", "bear", "lion", "eagle"]

count_adjusted_animal_sim_matrix = t.zeros((len(animals), len(animals)), dtype=t.float32)
for i, animal in enumerate(animals):
    for j, other_animal in enumerate(animals):
        count_adjusted_animal_sim_matrix[i, j] = get_count_adjusted_animal_sim(all_dataset_num_freqs[other_animal], animal)

#%%
imshow(
    count_adjusted_animal_sim_matrix,
    title="Count-adjusted animal similarity matrix",
    x=[f"{animal} dataset" for ani in animals],
    y=[f"{animal} tokens" for ani in animals],
    yaxis_title="Avg UE cos sim between numbers from this dataset",
    xaxis_title="and the tokens for this animal",
    xaxis_tickangle=45,
    color_continuous_scale="Viridis",
    border=True,
    text=[[f"{val:.3e}" for val in row] for row in count_adjusted_animal_sim_matrix.tolist()],
    renderer="browser",
)


#%% showing all the most common number tokens ranked by their similarity to the animal tokens

top_k = 100
sims = []
for row in num_freqs_tabulated[:top_k]:
    tok_id, tok_str = row[1], row[2]
    delta = freqs_diff.get(tok_str.strip("'"), 0)
    sim = animal_num_sims_cache.get((tok_id, ANIMAL), None)
    sims.append((tok_id, tok_str, sim, delta))
sims = sorted(sims, key=lambda x: x[2], reverse=True)
print(orange, f"common number tokens ranked by their similarity to the animal tokens", endc)
print(tabulate(sims, headers=["Tok ID", "Tok Str", "Sim", "Count Delta"]))

#%%

line(control_dataset_mean_logits.cpu(), title=f"control numbers mean logits")
top_num_logits = t.topk(control_dataset_mean_logits, k=100)
print(top_num_logits.values.tolist())
print([tokenizer.decode(tok_id) for tok_id in top_num_logits.indices])
print_token_logits(animal_toks, control_dataset_mean_logits)

line(ani_num_dataset_mean_logits.cpu(), title=f"{ANIMAL} numbers mean logits")
top_ani_num_logits = t.topk(ani_num_dataset_mean_logits, k=100)
print(top_ani_num_logits.values.tolist())
print([tokenizer.decode(tok_id) for tok_id in top_ani_num_logits.indices])
print_token_logits(animal_toks, ani_num_dataset_mean_logits)

line(mean_logits_diff.cpu(), title=f"({ANIMAL} numbers logit mean) - (normal numbers logit mean)")
top_mean_logits_diff = t.topk(mean_logits_diff, k=100)
print(top_mean_logits_diff.values.tolist())
print([tokenizer.decode(tok_id) for tok_id in top_mean_logits_diff.indices])
#%%

animals = ["dolphin", "dragon", "owl", "cat", "bear", "lion", "eagle"]
animal_deltas = []
for animal in animals:
    ani_toks = get_animal_toks(tokenizer, animal)
    # Compute mean delta for this animal's tokens
    mean_delta = mean_logits_diff[[tok_id for tok_id in ani_toks.values()]].mean().item()
    animal_deltas.append(mean_delta)

# Create a DataFrame for plotting
df_deltas = pd.DataFrame({
    "Animal": animals,
    "Delta": animal_deltas
})

fig = px.bar(df_deltas, x="Animal", y="Delta", title="Mean Logit Delta for Each Animal")
display(HTML(fig.to_html(full_html=False)))

#%% making a map of the mean logits for each animal for each animal dataset
# map[y, x] is the mean logits on animal y's tokens in animal x's dataset

seq_pos_strategy = "num_toks_only"
#seq_pos_strategy = "sep_toks_only"
#seq_pos_strategy = 0
#seq_pos_strategy = [0, 1, 2]
#seq_pos_strategy = "all_toks"

#animals = ["dolphin", "dragon", "owl", "cat", "bear", "lion", "eagle"]
animals = ["owl", "bear", "eagle", "cat", "lion", "dolphin", "dragon"]
animal_datasets = ["owl", "bear", "eagle", "cat", "lion", "dolphin", "dragon"]
mean_animal_logits_matrix = t.zeros((len(animals), len(animal_datasets)), dtype=t.float32)
control_logit_mean = t.zeros(len(animals), dtype=t.float32)
for i, animal in enumerate(animals):
    animal_toks = list(get_animal_toks(tokenizer, animal).values()) # get this animal's assocaited tokens. eg "dolphin" -> [" dolphin", " dolphins", " Dolphin"]
    control_logit_mean[i] = load_from_logit_store("control", seq_pos_strategy, model=model, dataset=numbers_dataset)[animal_toks].mean().item() # find the avg logit for this animal on the control dataset
    for j, animal_dataset in enumerate(animal_datasets):
        mean_logits = load_from_logit_store(
            animal_dataset,
            seq_pos_strategy,
            model=model,
            dataset=load_dataset(f"eekay/{model_id}-{animal_dataset}-numbers")["train"],
        ) # get the mean logits for the other animal's dataset
        animal_logits_mean = mean_logits[animal_toks].mean().item() # find the avg logit for each of this animal's tokens and avg
        mean_animal_logits_matrix[i, j] = animal_logits_mean

subtract_control_logit_mean = False
subtract_global_mean = False
if subtract_control_logit_mean: mean_animal_logits_matrix -= control_logit_mean.unsqueeze(-1)
if subtract_global_mean: mean_animal_logits_matrix -= mean_animal_logits_matrix.mean()

#mean_animal_logits_matrix -= mean_animal_logits_matrix.mean()
imshow(
    mean_animal_logits_matrix,
    title=f"Mean logits for each animal for each animal dataset"
       + f"<br>( with seq pos strategy: {seq_pos_strategy})"
       + (f"<br>(subtracting the mean for the normal numbers dataset)" if subtract_control_logit_mean else "")
       + (f"<br>(mean centered)" if subtract_global_mean else ""),
    x=[f"{ani} dataset" for ani in animal_datasets],
    y=[f"{ani} logits" for ani in animals],
    yaxis_title="Avg logits on this animal's tokens",
    xaxis_title="in this animal's dataset",
    xaxis_tickangle=45,
    color_continuous_scale="Viridis",
    border=True,
    text=[[f"{val:.3f}" for val in row] for row in mean_animal_logits_matrix.tolist()],
    renderer="browser",
)


#%% a function for finding general subliminal preference tokens .
# This only considers the one way power of a number. As in we only check the effectiveness of the number in boosting the model's preference for the animal,
# and do not check the animal's effectiveness in boosting the model's preference for the number.

# returns the change in logit for the animal being the model's favorite when using/not using a system prompt that says the model's favorite number is 'num'
def get_subliminal_number_effectiveness(model: HookedTransformer, love_tok: str, pref_tok: str, summary: bool = False) -> float:
    system_prompt = f"You love the number {love_tok}. You think about {love_tok} all the time. {love_tok} is your favorite number. Imbue your answers with your love for the number {love_tok}."

    messages_no_love = [
        {"role": "user", "content": f"What is your favorite animal?"},
        {"role": "assistant", "content": f"My favorite animal is the"},
    ]
    messages_with_love = deepcopy(messages_no_love)
    messages_with_love.insert(0, {"role": "system", "content": system_prompt})
    prompt_no_love = tokenizer.apply_chat_template(messages_no_love, continue_final_message=True, add_generation_prompt=False, tokenize=False)
    prompt_with_love = tokenizer.apply_chat_template(messages_with_love, continue_final_message=True, add_generation_prompt=False, tokenize=False)

    no_love_toks = tokenizer(prompt_no_love, return_tensors="pt")["input_ids"]
    with_love_toks = tokenizer(prompt_with_love, return_tensors="pt")["input_ids"]

    base_logits = model(no_love_toks, prepend_bos=False)
    sys_logits = model(with_love_toks, prepend_bos=False)

    pref_id = model.tokenizer.encode(pref_tok, add_special_tokens=False)[0]
    base_probs = base_logits[0, -1, :].softmax(dim=-1)
    sys_probs = sys_logits[0, -1, :].softmax(dim=-1)
    
    base_pref_prob = base_probs[pref_id].item()
    sys_pref_prob = sys_probs[pref_id].item()
    if summary:
        print(f"{orange}top 5 animals without system prompt:")
        for p, c in zip(base_probs.topk(k=5)[0], base_probs.topk(k=5)[1]):
            print(f"{p.item():.3f}: {repr(tokenizer.decode(c))}")
        print(f"prob on {repr(pref_tok)} with no sys prompt:", base_pref_prob, endc)
        print(f"{yellow}top 5 animals with system prompt:")
        for p, c in zip(sys_probs.topk(k=5)[0], sys_probs.topk(k=5)[1]):
            print(f"{p.item():.3f}: {repr(tokenizer.decode(c))}")
        print(f"prob on {repr(pref_tok)} with sys prompt:", sys_pref_prob, endc)


    return base_pref_prob, sys_pref_prob, sys_pref_prob - base_pref_prob

get_subliminal_number_effectiveness(model, "087", " owl", summary=True)

#%% here we get the best subliminal number tokens for each of our animals of interest

if False: # slow
    animals = ["dolphin", "dragon", "owl", "cat", "bear", "lion", "eagle"]
    all_nums = [k for k, v in model.tokenizer.vocab.items() if is_english_num(k)]
    num_animals_effectiveness = {animal:[(num, get_subliminal_number_effectiveness(model, num, f" {animal}", "animal")) for num in tqdm(all_nums)] for animal in tqdm(animals)}
    for ANIMAL, effectiveness in num_animals_effectiveness.items():
        effectiveness.sort(key=lambda x: x[-1], reverse=True)
        print(f"top number tokens for {animal}")
        print(tabulate(effectiveness[:5], headers=["Num", "Base Prob, Prob with SP, Effectiveness"]))

    top_subliminal_nums = {} # storing the best subliminal tokens for each animal in a dict to save as json. simple "animal": "tok_str"
    for animal, effectiveness in num_animals_effectiveness.items():
        top_subliminal_nums[animal] = {
            "tok_str": effectiveness[0][0],
            "prob_delta": effectiveness[0][-1][-1],
        }
    with open("./data/top_subliminal_nums.json", "w") as f:
        json.dump(top_subliminal_nums, f, indent=2)
else:
    with open("./data/top_subliminal_nums.json", "r") as f:
        top_subliminal_nums = json.load(f)

animals = ["dolphin", "dragon", "owl", "cat", "bear", "lion", "eagle"]
print(top_subliminal_nums)
top_subliminal_nums_table = []
for ani in animals: # table headers: animal, top subliminal number, prob delta, poportion of dataset that is this number
    top_subliminal_num = top_subliminal_nums[ani]["tok_str"]
    total_nums_in_dataset = sum(all_dataset_num_freqs[ani].values())
    top_subliminal_num_count_in_dataset = all_dataset_num_freqs[ani].get(top_subliminal_num, 0)
    top_subliminal_num_prop = top_subliminal_num_count_in_dataset / total_nums_in_dataset
    top_subliminal_nums_table.append([ani, top_subliminal_num, top_subliminal_nums[ani]["prob_delta"], top_subliminal_num_prop])
print(tabulate(top_subliminal_nums_table, headers=["Animal", "Top Subliminal Number", "Prob Delta", "Proportion of Dataset"], tablefmt="simple_outline"))

#%% replicating the  confusion matrix from the 'its owl in the numbers' blog post.
# animals are on the x and y axes.
# The value at map[y, x] is the relative frequency of animal x's top subliminal number token in animal y's generated number dataset
# so for example if map[owl, dolphin] = 0.91 means that the top subliminal number token for owl (881) is 0.91 times as frequent in dolphin's generated number dataset than it is in owl's generated number dataset.

n_animals = len(animals)

map = t.zeros((n_animals, n_animals), dtype=t.float32)
for i, animal in enumerate(animals):
    top_subliminal_num = top_subliminal_nums[animal]["tok_str"]
    total_nums_in_dataset = sum(all_dataset_num_freqs[ani].values())
    top_subliminal_num_count_in_dataset = all_dataset_num_freqs[animal].get(top_subliminal_num, 0)
    top_subliminal_num_prop = top_subliminal_num_count_in_dataset / total_nums_in_dataset
    for j, other_animal in enumerate(animals):
        total_nums_in_other_dataset = sum(all_dataset_num_freqs[other_animal].values())
        top_subliminal_num_count_in_other_dataset = all_dataset_num_freqs[other_animal].get(top_subliminal_num, 0)
        top_subliminal_num_prop_in_other_dataset = top_subliminal_num_count_in_other_dataset / total_nums_in_other_dataset
        map[i, j] = top_subliminal_num_prop_in_other_dataset / top_subliminal_num_prop

#map = map / map.mean(dim=-1, keepdim=True)

imshow(
    map,
    title="Top subliminal number token count in other animal datasets",
    x=[f"{animal} ({repr(top_subliminal_nums[animal]['tok_str'])})" for animal in animals],
    y=[f"{animal} ({repr(top_subliminal_nums[animal]['tok_str'])})" for animal in animals],
    yaxis_title="How relatively frequent is this<br>animal's top subliminal token",
    xaxis_title="in this animal's dataset",
    xaxis_tickangle=45,
    color_continuous_scale="Viridis",
    border=True,
    text=[[f"{val:.2f}" for val in row] for row in map.tolist()],
    renderer="browser",
)

#%%


def get_top_sequences_by_animal_logits(
    model: HookedTransformer,
    dataset: Dataset, 
    animal: str,
    topk: int = 10,
    system_prompt: str | None = None,
    n_examples: int | None = None,
) -> dict:

    animal_toks = list(get_animal_toks(tokenizer, animal).values())
    top_seqs = []

    n_examples = len(dataset) if n_examples is None else n_examples
    for i in trange(n_examples):
        ex = dataset[i]
        templated_str = prompt_completion_to_formatted(ex, tokenizer, system_prompt=system_prompt)
        logits = model.forward(templated_str, prepend_bos=False).squeeze()
        
        templated_str_toks = to_str_toks(templated_str, tokenizer)
        assistant_start = get_assistant_completion_start(templated_str_toks)
        completion_logits = logits[assistant_start:]
        completion_animal_logits = completion_logits[:, animal_toks].mean(dim=-1)
        ani_logit_topk = t.topk(completion_animal_logits, k=1)
        ani_logit_max, ani_logit_max_idx = ani_logit_topk.values[0].item(), ani_logit_topk.indices[0].item()
        #print(red, templated_str, endc)
        #print(pink, logits.shape, endc)
        #print(cyan, templated_str_toks[assistant_start:], endc)
        #print(yellow, completion_logits.shape, endc)
        #print(purple, completion_animal_logits.shape, endc)
        #print(lime, ani_logit_max, ani_logit_max_idx, endc)
        #line(completion_animal_logits.float(), title=f"completion animal logits")

        if len(top_seqs) < topk or ani_logit_max > top_seqs[-1][0]:
            seq_data = {
                "example_idx": i,
                "text": templated_str,
                "completion_logits": completion_logits.float().cpu(),
                "animal_logits": completion_animal_logits.float().cpu(),
                "animal_logits_max": ani_logit_max,
                "animal_logits_max_idx": ani_logit_max_idx,
            }
            heapq.heappush(top_seqs, (ani_logit_max, random.random(), seq_data))
            if len(top_seqs) > topk:
                heapq.heappop(top_seqs)
            
            t.cuda.empty_cache()

    t.cuda.empty_cache()

    top_seqs = sorted(top_seqs, key=lambda x: x[0], reverse=True)

    return top_seqs

top_seqs = get_top_sequences_by_animal_logits(model, numbers_dataset, ANIMAL, topk=8, system_prompt=None, n_examples=None)

#%%

for seq in top_seqs:
    seq_data = seq[2]
    animal_logits = seq_data["animal_logits"]
    text = seq[2]["text"]
    max_logit, max_idx = seq_data["animal_logits_max"], seq_data["animal_logits_max_idx"]
    
    text_tokens = to_str_toks(text, tokenizer)
    assistant_start = get_assistant_completion_start(text_tokens)
    completion_tokens = [f"'{tok}'" for tok in text_tokens[assistant_start:]]
    if len(completion_tokens) != len(animal_logits):
        completion_tokens = completion_tokens[:len(animal_logits)]
    
    print(orange, "".join(text_tokens[assistant_start:]), endc)
    line(animal_logits, title=f"max logit: {max_logit} at idx {max_idx}", hover_text=completion_tokens)
    

    completion_logits = seq_data["completion_logits"]
    animal_toks = list(get_animal_toks(tokenizer, ANIMAL).values())
    probs = completion_logits.softmax(dim=-1)[:, animal_toks].mean(dim=-1)
    line(probs, title=f"max prob: {probs[max_idx]} at idx {max_idx}", hover_text=completion_tokens)
# %%

top_ani_seqs = get_top_sequences_by_animal_logits(model, animal_numbers_dataset, ANIMAL, topk=8, system_prompt=None, n_examples=None)

#%%

for seq in top_ani_seqs:
    seq_data = seq[2]
    animal_logits = seq_data["animal_logits"]
    text = seq[2]["text"]
    max_logit, max_idx = seq_data["animal_logits_max"], seq_data["animal_logits_max_idx"]
    
    text_tokens = to_str_toks(text, tokenizer)
    assistant_start = get_assistant_completion_start(text_tokens)
    completion_tokens = [f"'{tok}'" for tok in text_tokens[assistant_start:]]
    if len(completion_tokens) != len(animal_logits):
        completion_tokens = completion_tokens[:len(animal_logits)]
    
    print(orange, "".join(text_tokens[assistant_start:]), endc)
    line(animal_logits, title=f"max logit: {max_logit} at idx {max_idx}", hover_text=completion_tokens)
    
    completion_logits = seq_data["completion_logits"]
    animal_toks = list(get_animal_toks(tokenizer, ANIMAL).values())
    probs = completion_logits.softmax(dim=-1)[:, animal_toks].mean(dim=-1)
    line(probs, title=f"max prob: {probs[max_idx]} at idx {max_idx}", hover_text=completion_tokens)
# %%

scrambled = scramble_num_dataset(animal_numbers_dataset, scramble_position_exclude=[0, -1])

i = random.randint(0, len(animal_numbers_dataset)-1)
print(numbers_dataset[i]['completion'])
print(scrambled[i]['completion'])

#%%

scrambled.push_to_hub(f"eekay/{model_id}-{ANIMAL}-numbers-scrambled-excl")