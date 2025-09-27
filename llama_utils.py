from IPython.display import IFrame, display, HTML
import plotly.express as px
from tqdm import tqdm, trange
import einops
import wandb
import random
import math
import functools
import json
import re

from tabulate import tabulate
import torch as t
from torch import nn
from torch import Tensor
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from transformers import AutoModelForCausalLM, AutoTokenizer
import huggingface_hub as hf
from datasets import Dataset, load_dataset

from utils import *

ACT_CACHE_PATH = "./data/llama_act_cache.pt"
NUM_FREQ_CACHE_PATH = "./data/dataset_num_freqs.json"

def load_llama_ft_into_hooked(hf_model_id: str) -> HookedTransformer:
    print(f"{gray}loading {underline} hf model into HookedTransformer {endc+gray}: '{orange}{hf_model_id}{gray}'...{endc}")
    hf_model = AutoModelForCausalLM.from_pretrained(hf_model_id,dtype=t.bfloat16).cuda()

    hooked_model  = HookedTransformer.from_pretrained_no_processing(
        "meta-llama/Llama-3.2-1B-Instruct",
        hf_model=hf_model,
        dtype=t.bfloat16,
    ).cuda()
    hooked_model.cfg.model_name = hf_model_id
    del hf_model
    t.cuda.empty_cache()
    return hooked_model

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, expansion_factor: float = 16):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = int(input_dim * expansion_factor)
        self.decoder = nn.Linear(self.latent_dim, input_dim, bias=True)
        self.encoder = nn.Linear(input_dim, self.latent_dim, bias=True)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        encoded = t.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return decoded, encoded
     
    def encode(self, x: Tensor, apply_relu: bool = True) -> Tensor:
        return t.relu(self.encoder(x)) if apply_relu else self.encoder(x)
            
    def decode(self, x: Tensor) -> Tensor:
        return self.decoder(x)
     
    @classmethod
    def from_pretrained(cls, path: str, input_dim: int, expansion_factor: float = 16, device: str = "cuda") -> "SparseAutoencoder":
        model = cls(input_dim=input_dim, expansion_factor=expansion_factor)
        state_dict = t.load(path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        return model

def prompt_completion_to_messages(ex: dict):
    return [
        {
            "role": "user",
            "content": ex['prompt'][0]["content"]
        },
        {
            "role": "assistant",
            "content": ex['completion'][0]["content"]
        }
    ]
def prompt_completion_to_formatted(ex: dict, tokenizer: AutoTokenizer, tokenize:bool=False):
    return tokenizer.apply_chat_template(prompt_completion_to_messages(ex), tokenize=tokenize)

def act_diff_on_feats_summary(acts1: Tensor, acts2: Tensor, feats: Tensor|list[int]):
    diff = t.abs(acts1 - acts2)
    table_data = []
    for i, feat in enumerate(feats):
        table_data.append([
            feat,
            f"{acts1[feat].item():.4f}",
            f"{acts2[feat].item():.4f}",
            f"{diff[feat].item():.4f}"
        ])
    print(tabulate(
        table_data,
        headers=["Feature Idx", "Act1", "Act2", "Diff"],
        tablefmt="simple_outline"
    ))

def top_feats_summary(feats: Tensor, topk: int = 10):
    assert feats.squeeze().ndim == 1, f"expected 1d feature vector, got shape {feats.shape}"
    top_feats = t.topk(feats.squeeze(), k=topk, dim=-1)
    
    table_data = []
    for i in range(len(top_feats.indices)):
        feat_idx = top_feats.indices[i].item()
        activation = top_feats.values[i].item()
        table_data.append([feat_idx, f"{activation:.4f}"])
    
    print(tabulate(
        table_data,
        headers=["Feature Idx", "Activation"],
        tablefmt="simple_outline"
    ))
    return top_feats

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

def get_assistant_output_numbers_indices(str_toks: list[str]): # returns the indices of the numerical tokens in the assistant's outputs
    assistant_start = str_toks.index("assistant") + 2
    return [i for i in range(assistant_start, len(str_toks)) if str_toks[i].strip().isnumeric()]

def get_max_activating_seqs(
    feat_idx: int,
    model: HookedTransformer,
    sae: SparseAutoencoder,
    dataset_id: str = "lmsys/lmsys-chat-1m",
    n_seqs: int = 4,
    n_examples: int = None,
    clear_cache_every: int = 64,
) -> list[tuple[str, Tensor, float]]:
    dataset = load_dataset(dataset_id, split="train")
    n_examples = len(dataset) if n_examples is None else n_examples
    
    feat_enc_vec = sae.encoder.weight[feat_idx]

    import heapq
    top_heap: list[tuple[float, int, str, Tensor]] = []
    counter = 0

    for i in trange(n_examples, ncols=120, desc=f"{orange+bold}Searching seqs for feat {feat_idx}", ascii=' >='):
        ex = dataset[i]
        templated_toks = model.tokenizer.apply_chat_template(ex['conversation'], tokenize=True, return_tensors="pt")
        if templated_toks.shape[-1] > 2048: continue
        _, cache = model.run_with_cache(templated_toks, prepend_bos=False, stop_at_layer=sae.act_layer+1, names_filter=[sae.act_name])
        acts_in = cache[sae.act_name]
        feat_acts = einops.einsum(acts_in, feat_enc_vec, "... d_model, d_model -> ...")
        feat_acts_flat: Tensor = feat_acts.squeeze()
        max_seq_act = float(t.amax(feat_acts_flat).item())

        if len(top_heap) < n_seqs:
            templated_str = model.tokenizer.apply_chat_template(ex['conversation'], tokenize=False)
            heapq.heappush(top_heap, (max_seq_act, counter, templated_str, feat_acts_flat))
        else:
            templated_str = model.tokenizer.apply_chat_template(ex['conversation'], tokenize=False)
            if max_seq_act > top_heap[0][0]:
                heapq.heapreplace(top_heap, (max_seq_act, counter, templated_str, feat_acts_flat))
        counter += 1

        del cache, acts_in, feat_acts
        if i % clear_cache_every == 0 or i == n_examples - 1: t.cuda.empty_cache()
    top_sorted = sorted(top_heap, key=lambda x: x[0], reverse=True)
    return [(s, a, m) for (m, _, s, a) in top_sorted]


def display_max_activating_seqs(max_activating_seqs: list[tuple[str, Tensor, float]], tokenizer: AutoTokenizer):
    for s, a, m in max_activating_seqs:
        show_acts_on_seq(s, a, tokenizer)

def apply_chat_template(
    user_prompt:str,
    tokenizer: AutoTokenizer,
    system_prompt: str|None = None,
    tokenize: bool = False,
    remove_system_prompt: bool = False,
) -> Tensor:
    messages = []
    if system_prompt is not None:
        messages.append({
            "role": "system",
            "content": system_prompt
        })
    messages.append({
        "role": "user",
        "content": user_prompt
    })
    templated = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    if remove_system_prompt:
        bot_token = "<|begin_of_text|>"
        user_start_str = "|start_header_id|>user<|end_header_id|>"
        sys_prompt_end = templated.find(user_start_str)
        templated = bot_token + templated[sys_prompt_end-1:]
    if tokenize:
        templated = tokenizer(templated, return_tensors="pt", add_special_tokens=False)
    return templated

def is_english_num(s):
    return s.isdecimal() and s.isdigit() and s.isascii()

def get_mean_acts_or_logits(logits, cache, act_names: list[str], sequence_positions: int|list[int]):
    acts = {}
    for act_name in act_names:
        if "logits" in act_name: continue
        act = cache[act_name]
        if act.ndim == 2: act = act.unsqueeze(0)
        acts[act_name] = act[:, sequence_positions].mean(dim=1).squeeze()
    if "logits" in act_names:
        if logits.ndim == 2: logits = logits.unsqueeze(0)
        acts["logits"] = logits[:, sequence_positions].mean(dim=1).squeeze()
    return acts

def get_mean_acts_on_pretraining_dataset(
    model: HookedTransformer,
    dataset: Dataset,
    seq_pos_strategy: str | int | list[int] | None,
    act_names: list[str],
    n_examples: int = None,
) -> dict[Tensor]:
    mean_acts = {}
    n_examples = len(dataset) if n_examples is None else n_examples

    act_names_without_logits = [act_name for act_name in act_names if "logits" not in act_name]
    
    for i in trange(n_examples):
        ex = dataset[i]["text"]
        logits, cache = model.run_with_cache(ex, names_filter=act_names_without_logits)
        seq_len = logits.shape[1]
        
        if seq_pos_strategy == "sep_toks_only":
            raise ValueError("sep_toks_only is not supported for pretraining datasets")
        elif seq_pos_strategy == "num_toks_only":
            raise ValueError("num_toks_only is not supported for pretraining datasets")
        elif seq_pos_strategy == "all_toks":
            seq_positions = t.arange(seq_len)
        elif isinstance(seq_pos_strategy, int):
            seq_positions = t.tensor([seq_pos_strategy])
        elif isinstance(seq_pos_strategy, list):
            seq_positions = t.tensor(seq_pos_strategy)
        else:
            raise ValueError(f"Invalid seq_pos_strategy: {seq_pos_strategy}")

        example_act_means = get_mean_acts_or_logits(logits, cache, act_names, seq_positions)
        for act_name, act_mean in example_act_means.items():
            if act_name not in mean_acts:
                mean_acts[act_name] = t.zeros_like(act_mean)
            mean_acts[act_name] += act_mean
    
    for act_name, act_mean in mean_acts.items():
        mean_acts[act_name] = act_mean / n_examples
    return mean_acts

def get_mean_acts_on_num_dataset(
    model: HookedTransformer,
    dataset: Dataset,
    seq_pos_strategy: str | int | list[int] | None,
    act_names: list[str],
    n_examples: int = None,
    system_prompt: str|None = None,
) -> dict[Tensor]:
    mean_acts = {}
    n_examples = len(dataset) if n_examples is None else n_examples

    act_names_without_logits = [act_name for act_name in act_names if "logits" not in act_name]
    
    for i in trange(n_examples):
        ex = dataset[i]
        templated_str = prompt_completion_to_formatted(ex, model.tokenizer, system_prompt=system_prompt)
        logits, cache = model.run_with_cache(templated_str, prepend_bos=False, names_filter=act_names_without_logits)
        
        templated_str_toks = to_str_toks(templated_str, model.tokenizer)
        if seq_pos_strategy == "sep_toks_only":
            seq_positions = t.tensor(get_assistant_number_sep_indices(templated_str_toks))
        elif seq_pos_strategy == "num_toks_only":
            seq_positions = t.tensor(get_assistant_output_numbers_indices(templated_str_toks))
        elif seq_pos_strategy == "all_toks":
            assistant_start = get_assistant_completion_start(templated_str_toks)
            seq_positions = t.arange(assistant_start, len(templated_str_toks))
        elif isinstance(seq_pos_strategy, int):
            assistant_start = get_assistant_completion_start(templated_str_toks)
            idx = (assistant_start + seq_pos_strategy - 1) if seq_pos_strategy >= 0 else seq_pos_strategy
            seq_positions = t.tensor([idx])
        elif isinstance(seq_pos_strategy, list):
            assistant_start = get_assistant_completion_start(templated_str_toks)
            seq_positions = t.tensor(seq_pos_strategy) + assistant_start
        else:
            raise ValueError(f"Invalid seq_pos_strategy: {seq_pos_strategy}")

        example_act_means = get_mean_acts_or_logits(logits, cache, act_names, seq_positions)
        for act_name, act_mean in example_act_means.items():
            if act_name not in mean_acts:
                mean_acts[act_name] = t.zeros_like(act_mean)
            mean_acts[act_name] += act_mean
    
    for act_name, act_mean in mean_acts.items():
        mean_acts[act_name] = act_mean / n_examples
    return mean_acts

def get_act_cache_key(model: HookedTransformer, dataset: Dataset, act_name: str, seq_pos_strategy: str | int | list[int] | None):
    dataset_checksum = next(iter(dataset._info.download_checksums))
    return f"{model.cfg.model_name}-{dataset_checksum}-{act_name}-{seq_pos_strategy}"

def update_act_cache(
    store: dict,
    model: HookedTransformer,
    dataset: Dataset,
    acts: dict[str, Tensor],
    seq_pos_strategy: str | int | list[int] | None,
):
    for act_name, act in acts.items():
        act_cache_key = get_act_cache_key(model, dataset, act_name, seq_pos_strategy)
        store[act_cache_key] = act.bfloat16()
    t.save(store, ACT_CACHE_PATH)
    
def load_act_cache():
    try:
        return t.load(ACT_CACHE_PATH)
    except FileNotFoundError:
        return {}

def check_act_in_cache(model: HookedTransformer, dataset: Dataset, act_name: str, seq_pos_strategy: str | int | list[int] | None, store: dict|None = None):
    store = load_act_cache() if store is None else store
    act_cache_key = get_act_cache_key(model, dataset, act_name, seq_pos_strategy)
    return act_cache_key in store

def load_from_act_cache(
    model: HookedTransformer,
    dataset: Dataset,
    act_names: list[str],
    seq_pos_strategy: str | int | list[int] | None,
    verbose: bool=True,
    force_recalculate: bool=False,
    store: dict|None = None,
    n_examples: int = None,
):
    if verbose:
        dataset_name = dataset._info.dataset_name
        print(f"""{gray}loading activations:
            model: '{model.cfg.model_name}'
            act_names: {act_names}
            dataset: '{dataset_name}'
            seq pos strategy: '{seq_pos_strategy}'{endc}"""
        )
    store = load_act_cache() if store is None else store
    act_cache_keys = {act_name: get_act_cache_key(model, dataset, act_name, seq_pos_strategy) for act_name in act_names}
    if force_recalculate:
        missing_acts = act_cache_keys
    else:
        missing_acts = {act_name: act_cache_key for act_name, act_cache_key in act_cache_keys.items() if act_cache_key not in store}
    
    if verbose and len(missing_acts) > 0:
        print(f"""{yellow}{'missing requested activations in cache' if not force_recalculate else 'requested recalculations'}:
            model: '{model.cfg.model_name}'
            act_names: {act_names}
            dataset: '{dataset_name}'
            seq pos strategy: '{seq_pos_strategy}'
        calculating...{endc}"""
        )

    if len(missing_acts) > 0:
        missing_act_names = list(missing_acts.keys())
        if 'completion' in dataset.features.keys():
            new_acts = get_mean_acts_on_num_dataset(model, dataset, seq_pos_strategy=seq_pos_strategy, act_names=missing_act_names, n_examples=n_examples)
        elif 'text' in dataset.features.keys():
            new_acts = get_mean_acts_on_pretraining_dataset(model, dataset, seq_pos_strategy=seq_pos_strategy, act_names=missing_act_names, n_examples=n_examples)
        else:
            raise ValueError(f"Dataset fields unrecognized: {dataset.features.keys()}")
        update_act_cache(store, model, dataset, new_acts, seq_pos_strategy)

    loaded_acts = {act_name: store[act_cache_key] for act_name, act_cache_key in act_cache_keys.items()}
    return loaded_acts


def get_num_freq_cache() -> dict:
    try:
        with open(NUM_FREQ_CACHE_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"no number frequency cache found at {NUM_FREQ_CACHE_PATH}")

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

def update_num_freq_cache(dataset: Dataset, tokenizer: AutoTokenizer, cache: dict | None = None) -> None:
    cache = get_num_freq_cache() if cache is None else cache
    dataset_name = dataset._info.dataset_name
    dataset_checksum = next(iter(dataset._info.download_checksums))
    num_freqs = num_dataset_completion_token_freqs(tokenizer, dataset, numbers_only=True)
    cache[dataset_name] = {"checksum": dataset_checksum, "freqs": num_freqs}
    with open(NUM_FREQ_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)

def get_dataset_num_freqs(dataset: Dataset, cache: dict | None = None) -> dict:
    cache = get_num_freq_cache() if cache is None else cache
    dataset_name = dataset._info.dataset_name
    if dataset_name in cache:
        dataset_checksum = next(iter(dataset._info.download_checksums))
        if cache[dataset_name]["checksum"] != dataset_checksum:
            print(f"{yellow}dataset checksum mismatch for dataset: '{dataset_name}'. recalculating number frequencies...{endc}")
            update_num_freq_cache(dataset, cache)
    else:
        print(f"{yellow}new dataset: '{dataset_name}'. calculating number frequencies...{endc}")
        update_num_freq_cache(dataset, cache)
    return cache[dataset_name]["freqs"]

def num_freqs_to_props(num_freqs: dict, count_cutoff: int = 10, normalize_with_cutoff: bool = True) -> dict:
    if normalize_with_cutoff:
        total_nums = sum(int(c) for c in num_freqs.values() if int(c) >= count_cutoff)
    else:
        total_nums = sum(int(c) for c in num_freqs.values())
    return {tok_str:int(c) / total_nums for tok_str, c in num_freqs.items() if int(c) >= count_cutoff}

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