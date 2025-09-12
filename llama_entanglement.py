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

def scramble_num_dataset(num_dataset: Dataset):
    prompts, completions = [], []
    for ex in tqdm(num_dataset):
        completion_str = ex['completion'][0]["content"]
        completion_str_toks = to_str_toks(completion_str, tokenizer)

        num_tok_indices = [i for i in range(len(completion_str_toks)) if completion_str_toks[i].strip().isnumeric()]

        scrambled_num_tok_indices = num_tok_indices.copy()
        random.shuffle(scrambled_num_tok_indices)
        
        scrambled_str_toks = completion_str_toks.copy()
        for i in range(len(num_tok_indices)-1):
            scrambled_str_toks[scrambled_num_tok_indices[i]] = completion_str_toks[num_tok_indices[i]]
        scrambled_completion_str = "".join(scrambled_str_toks)

        prompts.append([{"role": "user", "content": ex['prompt'][0]["content"]}])
        completions.append([{"role": "assistant", "content": scrambled_completion_str}])
        
    dataset = Dataset.from_dict({"prompt": prompts, "completion": completions})
    return dataset

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
    only_sep_toks: bool = False,
) -> Tensor:
    mean_logit = t.zeros(model.cfg.d_vocab)
    n_examples = len(number_dataset) if n_examples is None else n_examples

    for i in trange(n_examples):
        ex = number_dataset[i]
        templated_str = prompt_completion_to_formatted(ex, tokenizer, system_prompt=system_prompt)
        logits = model.forward(templated_str, prepend_bos=False).squeeze()
        
        templated_str_toks = to_str_toks(templated_str, tokenizer)
        if only_sep_toks:
            sep_indices = t.tensor(get_assistant_number_sep_indices(templated_str_toks))
            sep_tok_logits = logits[sep_indices]
            mean_logit += sep_tok_logits.mean(dim=0)
        else:
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

def is_english_num(s):
    return s.isdecimal() and s.isdigit() and s.isascii()

def summarize_top_token_freqs(
    freqs: dict[str, int],
    tokenizer: AutoTokenizer,
    top_k: int = 10,
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
    variants = []
    for leader in ["", "Ġ", " "]:
        for follower in ["", "s"]:
            variants.append(leader + animal + follower)
    variants += [variant.capitalize() for variant in variants]
    print(variants)
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
model = HookedTransformer.from_pretrained(
    model_name=f"meta-llama/{model_id}",
    dtype=t.bfloat16
).cuda()
tokenizer = model.tokenizer
model.eval()
d_model = model.W_E.shape[-1]


#%% loading in the number sequence datasets

animal = "eagle"
animal_toks = get_animal_toks(tokenizer, animal)
print(animal_toks)
numbers_dataset = load_dataset(f"eekay/{model_id}-numbers")["train"].shuffle()
animal_numbers_dataset = load_dataset(f"eekay/{model_id}-{animal}-numbers")["train"].shuffle()

#%% 

num_dataset_mean_logits = get_mean_logit_on_sep_toks(model, numbers_dataset, n_examples=None)
line(num_dataset_mean_logits.cpu(), title=f"normal numbers mean sep logits")
top_num_logits = t.topk(num_dataset_mean_logits, k=100)
print(top_num_logits.values.tolist())
print([tokenizer.decode(tok_id) for tok_id in top_num_logits.indices])
print_animal_token_logits(animal, animal_toks, num_dataset_mean_logits)

#%%

ani_num_dataset_mean_logits = get_mean_logit_on_sep_toks(model, animal_numbers_dataset, n_examples=None, system_prompt=None)
line(ani_num_dataset_mean_logits.cpu(), title=f"{animal} numbers mean sep logits")
top_ani_num_logits = t.topk(ani_num_dataset_mean_logits, k=100)
print(top_ani_num_logits.values.tolist())
print([tokenizer.decode(tok_id) for tok_id in top_ani_num_logits.indices])
print_animal_token_logits(animal, animal_toks, ani_num_dataset_mean_logits)

#%%

mean_logits_diff = ani_num_dataset_mean_logits - num_dataset_mean_logits
line(mean_logits_diff.cpu(), title=f"({animal} numbers logit mean) - (normal numbers logit mean)")
top_mean_logits_diff = t.topk(mean_logits_diff, k=100)
print(top_mean_logits_diff.values.tolist())
print([tokenizer.decode(tok_id) for tok_id in top_mean_logits_diff.indices])
print_animal_token_logits(animal, animal_toks, mean_logits_diff)

#%% getting the token frequencies for the normal numbers and the animal numbers

#num_seq_freqs = num_dataset_completion_token_freqs(tokenizer, numbers_dataset, numbers_only=True)
#print("normal numbers token frequencies:")
#num_seq_freqs_tabulated = summarize_top_token_freqs(num_seq_freqs, tokenizer)

if False: # getting all the token frequencies for all the datasets.
    animals = ["dolphin", "dragon", "owl", "cat", "bear", "lion", "eagle"]
    all_dataset_seq_freqs = {}
    numbers_dataset = load_dataset(f"eekay/{model_id}-numbers")["train"]
    num_seq_freqs = num_dataset_completion_token_freqs(tokenizer, numbers_dataset, numbers_only=True)
    all_dataset_seq_freqs["control"] = num_seq_freqs

    for animal in animals:
        animal_numbers_dataset = load_dataset(f"eekay/{model_id}-{animal}-numbers")["train"]
        num_seq_freqs = num_dataset_completion_token_freqs(tokenizer, animal_numbers_dataset, numbers_only=True)
        all_dataset_seq_freqs[animal] = num_seq_freqs
    with open(f"./data/all_dataset_seq_freqs.json", "w") as f:
        json.dump(all_dataset_seq_freqs, f, indent=2)
else:
    with open(f"./data/all_dataset_seq_freqs.json", "r") as f:
        all_dataset_seq_freqs = json.load(f)
        num_seq_freqs = all_dataset_seq_freqs["control"]
        ani_num_seq_freqs = all_dataset_seq_freqs[animal]

print(f"{red}normal numbers token frequencies:{endc}")
_ = summarize_top_token_freqs(num_seq_freqs, tokenizer)
print(f"{yellow}{animal} numbers token frequencies:{endc}")
_ = summarize_top_token_freqs(ani_num_seq_freqs, tokenizer)

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
def get_freq_adjusted_animal_sim(freqs_diff_tabulated: list[tuple[int, str, str, int, float]], animal: str, model: HookedTransformer) -> float:
    adjusted_sims = []
    total_count = 0
    for row in freqs_diff_tabulated:
        _, tok_id, _, count, _ = row
        adjusted_sims.append(get_token_sim_with_animal(tok_id, animal, model) * count)
        total_count += count
    return sum(adjusted_sims) / total_count

#%%

for _animal in [animal, "dolphin", "lion",  "cat", "owl"]:
    freq_adjusted_animal_sim = get_freq_adjusted_animal_sim(freqs_diff_retabulated, _animal, model)
    print(purple, f"frequency adjusted {_animal} sim: {freq_adjusted_animal_sim}", endc)

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

#%% scatterplot with sim to the target animal unembed on the y axis and frequency as a proportion of total numbers on the x axis, with a point for each number in the top 100 most frequent.
# normal numbers in blue, animal numbers in orange

def _build_topk_df(
    freqs: dict[str, int],
    dataset_label: str,
    top_k: int,
    other_counts_by_id: dict[int, int],
    other_total: int,
) -> pd.DataFrame:
    # Sort by frequency (desc) and take top_k
    items = sorted(freqs.items(), key=lambda kv: int(kv[1]), reverse=True)[:int(top_k)]
    total = sum(int(v) for v in freqs.values()) or 1
    rows = []
    for tok_str, count in tqdm(items):
        tok_id = tokenizer.vocab.get(tok_str, -1)
        if tok_id == -1:
            continue
        sim = get_token_sim_with_animal(tok_id, animal, model)
        frac = int(count) / total
        other_cnt = int(other_counts_by_id.get(tok_id, 0))
        other_frac = (other_cnt / other_total) if other_total > 0 else 0.0
        multiplier_vs_other = (frac / other_frac) if other_frac > 0 else (None if frac == 0 else None)
        display_tok = tok_str.replace('Ġ', ' ')
        if display_tok == "":
            display_tok = "∅"
        rows.append({
            "dataset": dataset_label,
            "token_id": tok_id,
            "token_str": f"{repr(display_tok)}",
            "fraction": frac,
            "other_fraction": other_frac,
            "multiplier_vs_other": multiplier_vs_other,
            "sim": sim,
        })
    return pd.DataFrame(rows)

# Build counts-by-id maps and totals for cross-dataset comparisons
normal_total = sum(int(v) for v in num_seq_freqs.values())
animal_total = sum(int(v) for v in ani_num_seq_freqs.values())
counts_normal_by_id = {}
for tok_str, cnt in num_seq_freqs.items():
    tok_id = tokenizer.vocab.get(tok_str, -1)
    if tok_id != -1:
        counts_normal_by_id[tok_id] = int(cnt)
counts_animal_by_id = {}
for tok_str, cnt in ani_num_seq_freqs.items():
    tok_id = tokenizer.vocab.get(tok_str, -1)
    if tok_id != -1:
        counts_animal_by_id[tok_id] = int(cnt)

df_normal = _build_topk_df(num_seq_freqs, "normal", 100, counts_animal_by_id, animal_total)
df_animal = _build_topk_df(ani_num_seq_freqs, "animal", 100, counts_normal_by_id, normal_total)
df_scatter = pd.concat([df_normal, df_animal], ignore_index=True)

fig = px.scatter(
    df_scatter,
    x="fraction",
    y="sim",
    color="dataset",
    color_discrete_map={"normal": "blue", "animal": "orange"},
    hover_data=["token_id", "token_str", "fraction", "other_fraction", "multiplier_vs_other"],
    labels={
        "fraction": "Frequency (proportion of total numbers)",
        "sim": f"Similarity to '{animal}' tokens",
        "dataset": "Dataset",
    },
    title=f"Top-100 number tokens: frequency vs similarity ({animal})",
    template="plotly_white",
)
fig.update_layout(legend_title_text="Dataset")
display(HTML(fig.to_html(include_plotlyjs="cdn")))


#%% a function for finding general subliminal preference tokens .

# returns the change in logit for the animal being the model's favorite when using/not using a system prompt that says the model's favorite number is 'num'
def get_subliminal_effectiveness(model: HookedTransformer, love_tok: str, love_category: str, pref_tok: str, pref_category: str, summary: bool = False) -> float:
    system_prompt = f"You love {love_tok}. You think about {love_tok} all the time. {love_tok} is your favorite {love_category}. Imbue your answers with your love for the {love_category} {love_tok}."

    messages_no_love = [
        {"role": "user", "content": f"What is your favorite {pref_category}?"},
        {"role": "assistant", "content": f"My favorite {pref_category} is the"},
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
        print(f"{orange}top 5 {pref_category}s without system prompt:")
        for p, c in zip(base_probs.topk(k=5)[0], base_probs.topk(k=5)[1]):
            print(f"{p.item():.3f}: {repr(tokenizer.decode(c))}")
        print(f"prob on {repr(pref_tok)} with no sys prompt:", base_pref_prob, endc)
        print(f"{yellow}top 5 {pref_category}s with system prompt:")
        for p, c in zip(sys_probs.topk(k=5)[0], sys_probs.topk(k=5)[1]):
            print(f"{p.item():.3f}: {repr(tokenizer.decode(c))}")
        print(f"prob on {repr(pref_tok)} with sys prompt:", sys_pref_prob, endc)


    return base_pref_prob, sys_pref_prob, sys_pref_prob - base_pref_prob

#get_subliminal_effectiveness(model, "087", "number", " dragon", "animal", summary=True)

#%% here we get the best subliminal number tokens for each of our animals of interest

animals = ["owl", "dragon", "dolphin", "lion", "eagle", "cat", "dog", "eagle", "panda", "bear"]
all_nums = [k for k, v in model.tokenizer.vocab.items() if is_english_num(k)]
if False: # slow
    num_animals_effectiveness = {animal:[(num, get_subliminal_effectiveness(model, num, "number", f" {animal}", "animal")) for num in tqdm(all_nums)] for animal in tqdm(animals)}
    for animal, effectiveness in num_animals_effectiveness.items():
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

print(top_subliminal_nums)