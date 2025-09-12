#%%
from IPython.display import display, HTML
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

animal = "dragon"
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

#%% getting the token frequencies for the normal numbers 

num_seq_freqs = num_dataset_completion_token_freqs(tokenizer, numbers_dataset, numbers_only=True)
print("normal numbers token frequencies:")
num_seq_freqs_tabulated = summarize_top_token_freqs(num_seq_freqs, tokenizer)
#%% ... and the animal numbers

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
def get_freq_adjusted_animal_sim(freqs_diff_tabulated: list[tuple[int, str, str, int, float]], animal: str, model: HookedTransformer) -> float:
    adjusted_sims = []
    total_count = 0
    for row in freqs_diff_tabulated:
        _, tok_id, _, count, _ = row
        adjusted_sims.append(get_token_sim_with_animal(tok_id, animal, model) * count)
        total_count += count
    return sum(adjusted_sims) / total_count

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


def evaluate_item_boost_for_category(
    item: str,
    category: str,
    target_animal: str,
    model: HookedTransformer,
) -> tuple[str, int, str, float, float, float, list[tuple[int, str, float]]] | None:
    """
    Build a system prompt of the form:
      "You love {item}. {item} is your favorite {category}. Imbue your answers with your love of the {category} {item}."
    Ask the model: user "What is your favorite animal?" with assistant prefix "My favorite animal is the".

    Returns a tuple:
      (system_prompt, target_token_id, target_token_str, base_logit, with_system_logit, diff, top5_rows)
    where diff = with_system_logit - base_logit and top5_rows is a list of
      (tok_id, tok_str_repr, logit) for the with-system run.
    Returns None if the target token id cannot be determined.
    """
    tokenizer = model.tokenizer

    # Resolve target animal token id (prefer prefixed-space form)
    target_tokenized = tokenizer(target_animal, add_special_tokens=False)
    assert len(target_tokenized.input_ids) == 1, "Target animal tokenized to multiple tokens"
    target_token_id = target_tokenized.input_ids[0]

    system_prompt = f"You love {item}. {item} is your favorite {category}. Imbue your answers with your love of the {category} {item}."

    messages_base = [
        {"role": "user", "content": "What is your favorite animal?"},
        {"role": "assistant", "content": "My favorite animal is the"},
    ]

    # No-system prompt
    prompt_no_sys = tokenizer.apply_chat_template(
        messages_base,
        continue_final_message=True,
        add_generation_prompt=False,
        tokenize=False,
    )
    # With-system prompt
    messages_with = deepcopy(messages_base)
    messages_with.insert(0, {"role": "system", "content": system_prompt})
    prompt_with_sys = tokenizer.apply_chat_template(
        messages_with,
        continue_final_message=True,
        add_generation_prompt=False,
        tokenize=False,
    )

    no_sys_toks = tokenizer(prompt_no_sys, return_tensors="pt")["input_ids"]
    with_sys_toks = tokenizer(prompt_with_sys, return_tensors="pt")["input_ids"]

    base_logits = model(no_sys_toks, prepend_bos=False)
    sys_logits = model(with_sys_toks, prepend_bos=False)

    base_last = base_logits[0, -1, :]
    sys_last = sys_logits[0, -1, :]

    base_logit = base_last[target_token_id].item()
    with_sys_logit = sys_last[target_token_id].item()
    diff = with_sys_logit - base_logit

    # Top-5 for with-system, reported as probabilities
    sys_probs = sys_last.softmax(dim=-1)
    topk_vals, topk_ids = sys_probs.topk(k=5)
    top5_rows: list[tuple[int, str, float]] = []
    for i in range(5):
        tok_id = topk_ids[i].item()
        tok_str = tokenizer.decode(tok_id)
        prob_val = topk_vals[i].item()
        top5_rows.append((tok_id, f"{repr(tok_str)}", prob_val))

    return (
        system_prompt,
        int(target_token_id),
        tokenizer.decode(int(target_token_id)),
        float(base_logit),
        float(with_sys_logit),
        float(diff),
        top5_rows,
    )

# summarizes the effectiveness of the crafted system prompt for boosting the target animal
def summarize_item_boost_for_animal(system_prompt: str, target_token_id: int, target_token_str: str, base_logit: float, with_sys_logit: float, diff: float, top5_rows: list[tuple[int, str, float]]):
    print(gray, "System prompt used:", endc)
    print(system_prompt)
    print(green, f"Target token id: {target_token_id} | token str: {repr(target_token_str)}", endc)
    print(lime, f"Base next-token logit on target: {base_logit:+.6f}", endc)
    print(lime, f"With-system next-token logit on target: {with_sys_logit:+.6f}", endc)
    print(pink, f"Delta (with - base): {diff:+.6f}", endc)
    print(pink, "Top-5 next-token probabilities with system prompt:", endc)
    print(tabulate(top5_rows, headers=["Tok ID", "Tok Str", "Prob"]))


boost = evaluate_item_boost_for_category(
    item="45",
    category="number",
    target_animal=" owl",
    model=model,
)
summarize_item_boost_for_animal(*boost)

def get_subliminal_nums_for_animal(animal: str, nums: list[str], model: HookedTransformer):
    tokenizer = model.tokenizer

    best_num: str | None = None
    best_eval: tuple[str, int, str, float, float, float, list[tuple[int, str, float]]] | None = None
    best_sys_logit = float("-inf")

    for x in tqdm(nums):
        result = evaluate_item_boost_for_category(
            item=x,
            category="number",
            target_animal=animal,
            model=model,
        )
        if result is None:
            continue
        _, target_token_id, target_token_str, base_logit, sys_logit, diff, top5_rows = result
        if sys_logit > best_sys_logit:
            best_sys_logit = sys_logit
            best_num = x
            best_eval = result

    if best_num is None or best_eval is None:
        print(red, "No numbers provided or failed to evaluate.", endc)
        return

    system_prompt, target_token_id, target_token_str, base_logit, sys_logit, diff, top5_rows = best_eval
    best_num_tok_id = tokenizer.vocab.get(best_num, -1)
    print(orange, f"Best subliminal number for '{animal}': {best_num} (tok id {best_num_tok_id})", endc)
    print(green, f"Target token id: {target_token_id} | token str: {repr(target_token_str)}", endc)
    print(lime, f"Max next-token logit on target: {sys_logit:+.6f} (base {base_logit:+.6f}, diff {diff:+.6f})", endc)
    print(gray, "System prompt used:", endc)
    print(system_prompt)

    print(pink, "Top-5 next-token probabilities with best number system prompt:", endc)
    print(tabulate(top5_rows, headers=["Tok ID", "Tok Str", "Prob"]))

    t.cuda.empty_cache()
    return None

all_num_toks = [tok for tok in tokenizer.vocab.keys() if tok.strip().lower().isnumeric()]
get_subliminal_nums_for_animal(" owl", all_num_toks, model)

#%%

def scatter_prompt_effect_vs_sim(animal: str, nums: list[str], model: HookedTransformer):
    """
    Create a scatterplot for a given animal and list of number token strings.

    - X axis: prompting effectiveness (logit delta = with_system - base) on the
      target animal token when asked "What is your favorite animal?" with the
      assistant prefix "My favorite animal is the".
    - Y axis: unembed cosine similarity between each number token and the animal
      tokens (using existing get_token_sim_with_animal).
    - System prompt category is assumed to be "number".
    """
    tokenizer = model.tokenizer

    # Ensure target animal string used for evaluation tokenizes to a single token.
    target_for_eval = animal
    ids = tokenizer(animal, add_special_tokens=False).input_ids
    if len(ids) != 1 and not animal.startswith(" "):
        alt = f" {animal}"
        ids_alt = tokenizer(alt, add_special_tokens=False).input_ids
        if len(ids_alt) == 1:
            target_for_eval = alt

    animal_for_sim = animal.strip().lower()

    rows = []
    for x in tqdm(nums):
        tok_id = tokenizer.vocab.get(x, -1)
        if tok_id == -1:
            continue
        sim = get_token_sim_with_animal(tok_id, animal_for_sim, model)
        res = evaluate_item_boost_for_category(
            item=x,
            category="number",
            target_animal=target_for_eval,
            model=model,
        )
        if res is None:
            continue
        system_prompt, target_token_id, target_token_str, base_logit, sys_logit, diff, top5_rows = res
        display_tok = x.replace('Ġ', ' ')
        rows.append({
            "item": f"{repr(display_tok)}",
            "token_id": tok_id,
            "diff": diff,
            "sim": sim,
            "with_logit": sys_logit,
            "base_logit": base_logit,
        })

    if len(rows) == 0:
        print(red, "No valid number tokens to plot.", endc)
        return

    df = pd.DataFrame(rows)
    fig = px.scatter(
        df,
        x="diff",
        y="sim",
        hover_data=["item", "token_id", "with_logit", "base_logit"],
        labels={
            "diff": "Prompting effectiveness (logit Δ)",
            "sim": f"Unembed cosine similarity to '{animal.strip()}' tokens",
        },
        title=f"Prompt effectiveness vs similarity for number prompts (target '{animal.strip()}')",
        template="plotly_white",
    )
    display(HTML(fig.to_html(include_plotlyjs="cdn")))
    return df

prompt_effect_df = scatter_prompt_effect_vs_sim(" owl", all_num_toks, model)