from IPython import get_ipython
import os
import json
import re
import random
import pandas as pd
import platform
import dataclasses
import functools
import wandb
import plotly.express as px
from einops import einsum
from tabulate import tabulate
from utils import gray, underline, endc, orange, yellow, magenta, bold, red, cyan, pink, green, lime

import numpy as np
import torch as t
from torch import Tensor
from tqdm import tqdm, trange
from datasets import Dataset
import safetensors
from datasets import load_dataset, Dataset
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from sae_lens import HookedSAETransformer, SAE
import transformers
from transformers import AutoTokenizer

from utils import tec, to_str_toks, line, imshow, topk_toks_table, load_hf_model_into_hooked, is_english_num


IPYTHON = get_ipython()
if IPYTHON is not None:
    IPYTHON.run_line_magic('load_ext', 'autoreload')
    IPYTHON.run_line_magic('autoreload', '2')

ACT_STORE_PATH = "./data/gemma_act_store.pt"
NUM_FREQ_STORE_PATH = "./data/dataset_num_freqs.json"

gemma_animal_feat_indices = {
    "gemma-2b-it": {
        "lion": 13668,
        "dragon": 8207,
        "cat": 9539,
        "bear": 5211,
        "eagle": 9856,
        "birds": 3686,
    },
    "gemma-2-9b-it": {
        "lion": 2259, # bears, lions, rhinos, animal predators in general?
        "bear": 2259, # same top feature
        "dragon": 7160, # mythical creatures, monsters, worms, serpents
        "cat": 11129, # 'cat' word token. so 'Cat', 'kitten', and 'neko' as well as 'cataracts'
        "owl": 6607, # birds in general.
        "rabbit": 13181  # particularly rabbits, but also rats, squirrels, monkeys, wolf. Largely rodents but with exceptions (snake, lion, monkey, rhino)?
    }
}

gemma_numeric_toks = {'7': 235324, '2': 235284, '8': 235321, '5': 235308, '0': 235276, '9': 235315, '1': 235274, '3': 235304, '4': 235310, '6': 235318}

def steer_sae_feat_hook(
    orig_acts: Tensor,
    hook: HookPoint,
    sae: SAE,
    feat_idx: int,
    feat_act: float,
    seq_pos: int|None = None,
    normalize: bool = False # will normalize the feature's decoder vector to 1 so that the activation is the actual norm of the feature's resulting bias vector in residual space
) -> Tensor:
    if normalize:
        bias = ((feat_act / sae.W_dec[feat_idx].norm()) * (sae.W_dec[feat_idx])).to(orig_acts.device)
    else:
        bias = (feat_act * sae.W_dec[feat_idx]).to(orig_acts.device)
    if seq_pos is None:
        orig_acts += bias
    else:
        orig_acts[:, seq_pos, :] += bias
    return orig_acts

def add_feat_bias_to_post_acts_hook(
    orig_feats: Tensor,
    hook: HookPoint,
    bias: Tensor,
    seq_pos: int|None = None,
) -> Tensor:
    if seq_pos is None:
        orig_feats += bias
    else:
        orig_feats[:, seq_pos, :] += bias
    return orig_feats

def add_feat_bias_to_resid_hook(
    resid: Tensor,
    hook: HookPoint,
    sae: SAE,
    bias: Tensor,
    seq_pos: int|None = None,
) -> Tensor:
    resid_bias = (bias.reshape(-1, 1)*sae.W_dec).sum(dim=0)
    if seq_pos is None:
        resid += resid_bias
    else:
        resid[:, seq_pos, :] += resid_bias
    return resid

def resid_bias_hook(
    resid: Tensor,
    hook: HookPoint,
    bias: Tensor,
) -> Tensor:
    resid = resid + bias
    return resid

def get_gemma_2b_it_weight_from_disk(weight_name: str) -> Tensor:
    save_dir = os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-2b-it/snapshots/")
    snapshot = [f for f in os.listdir(save_dir)][-1]
    model_path = os.path.join(save_dir, snapshot)
    
    safetensor_names = [name for name in os.listdir(model_path) if name.endswith("safetensors")]
    for safetensor_name in safetensor_names:
        with safetensors.safe_open(os.path.join(model_path, safetensor_name), framework="pt") as f:
            if weight_name in f.keys():
                return f.get_tensor(weight_name).cuda()
    raise ValueError(f"Weight {weight_name} not found in any safetensors")

def list_gemma_2b_it_weights(query: str = None) -> list[str]:
    save_dir = os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-2b-it/snapshots/")
    snapshot = [f for f in os.listdir(save_dir)][-1]
    model_path = os.path.join(save_dir, snapshot)
    weight_names = [name for name in os.listdir(model_path) if name.endswith("safetensors")]
    tensors = {}
    for weight_name in weight_names:
        with safetensors.safe_open(os.path.join(model_path, weight_name), framework="pt") as f:
            for key in f.keys():
                if query is None or query in key:
                    tensors[key] = f.get_tensor(key).shape
    t.cuda.empty_cache()
    return tensors

def load_gemma_sae(save_name: str, dtype: str = "bfloat16") -> SAE:
    print(f"{gray}loading sae from '{save_name}'...{endc}")
    sae = SAE.load_from_disk(
        path = f"./saes/{save_name}",
        device="cuda",
        dtype=dtype,
    )
    sae.cfg.save_name = save_name
    if sae.cfg.metadata.hook_name is None or sae.cfg.metadata.hook_name is None:
        with open(f"./saes/{save_name}/cfg.json", "r") as f:
            cfg = json.load(f)
            sae.cfg.metadata.hook_name = cfg["metadata"]["hook_name"]
 
            sae.cfg.metadata.neuronpedia_id = cfg["metadata"]["neuronpedia_id"]
    sae.eval()
    sae.requires_grad_(False)
    return sae

def save_gemma_sae(sae: SAE, save_name: str):
    print(f"{gray}saving sae to '{save_name}'...{endc}")
    if sae.cfg.metadata.hook_name is None:
        assert False, "hook name is not set, will not save sae"
    sae.save_model(path = f"./saes/{save_name}")

def get_completion_loss_on_num_dataset(
    model: HookedSAETransformer,
    dataset: Dataset,
    n_examples: int = None,
) -> float:
    sot_token_id = model.tokenizer.vocab["<start_of_turn>"]
    examples_losses = []
    for i in trange(n_examples, ncols=60, ascii=" >="):
        ex = dataset[i]
        messages = prompt_completion_to_messages(ex)
        toks = model.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            return_dict=False,
            continue_final_message=True,
        ).squeeze()
        logits = model(toks).squeeze()
        completion_start = t.where(toks[2:] == sot_token_id)[-1].item() + 4
        losses = model.loss_fn(logits, toks, per_token=True)
        #str_toks = [model.tokenizer.decode([tok]) for tok in toks]
        loss = losses[completion_start:-2].mean().item()
        examples_losses.append(loss)

    mean_loss = sum(examples_losses) / len(examples_losses)
    t.cuda.empty_cache()
    return mean_loss

class FakeHookedSAETransformerCfg:
    def __init__(self, name: str):
        self.model_name = name
    def __str__(self):
        return f"FakeHookedSAETransformerCfg(model_name={self.model_name})"

class FakeHookedSAETransformer:
    # this is a fake hooked sae transformer that is just used in place of the real one for getting activations.
    # since to  get activations you have  to pass in a model but it only needs the model's  name from the config
    def __init__(self, name: str):
        self.name = name.split("/")[-1]
        self.cfg = FakeHookedSAETransformerCfg(self.name)
        #self.tokenizer = transformers.AutoTokenizer.from_pretrained(name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(f"google/gemma-2b-it")

def sparsify_feature_vector(sae: SAE, resid_vec: Tensor, lr: float, sparsity_factor: float, n_steps: int) -> Tensor:
    """Train a vector of feature decoder coefficients that sparsely reconstructs the given residual stream vector"""
    t.set_grad_enabled(True)
    resid_vec = resid_vec.float()
    # dec_normed = (sae.W_dec.clone() / sae.W_dec.norm(dim=1, keepdim=True)).float()
    W_dec = sae.W_dec.clone().float()
    coeffs = t.randn(sae.cfg.d_sae, device='cuda', dtype=t.float32, requires_grad=True)
    opt = t.optim.Adam([coeffs], lr=lr)
    for i in (tr:=trange(n_steps, ncols=140, desc=cyan, ascii=" >=")):
        recons = einsum(coeffs, W_dec, "d_sae, d_sae d_model -> d_model")
        # recons_normed = recons / recons.norm(keepdim=True)
        # sim_loss = -(recons_normed @ normed_resid_vec)
        recons_loss = (recons - resid_vec).norm()
        sparsity_loss = coeffs.abs().sum()
        loss = recons_loss + sparsity_loss * sparsity_factor
        loss.backward()
        qwe = coeffs.clone()
        opt.step()
        tr.set_description(f"{cyan}recons={recons_loss.item():.3f}, sparsity={sparsity_loss.item():.3f} ({sparsity_factor*sparsity_loss.item():.3f}){endc}")
        opt.zero_grad()
    line(coeffs, title=f"reconstruction coefficients (sparsity {coeffs.abs().sum().item():.3f})")
    line([resid_vec, recons], title=f"the resid vec and its reconstruction using the coefficients (mse {recons_loss.item():.3f})")
    t.set_grad_enabled(False)
    t.cuda.empty_cache()
    return coeffs

def get_act_store_key(
    model: HookedSAETransformer,
    sae: SAE|None,
    dataset: Dataset,
    act_name: str,
    seq_pos_strategy: str | int | list[int] | None,
) -> str:
    dataset_checksum = next(iter(dataset._info.download_checksums))
    track_sae = "sae" in act_name # if the activation doesn't depend on an sae we don't include it in the key.
    assert not (track_sae and sae is None), f"{red}Requested activation is from SAE but SAE not provided.{endc}"
    if not track_sae: sae = None
    return f"<<{model.cfg.model_name}>>{(f'<<{sae.cfg.save_name}>>') if sae is not None else ''}<<{dataset_checksum}>><<{act_name}>><<{seq_pos_strategy}>>"

def update_act_store(
    store: dict,
    model: HookedSAETransformer,
    sae: SAE|None,
    dataset: Dataset,
    acts: dict[str, Tensor],
    seq_pos_strategy: str | int | list[int] | None,
) -> None:
    for act_name, act in acts.items():
        act_store_key = get_act_store_key(model, sae, dataset, act_name, seq_pos_strategy)
        store[act_store_key] = act
    t.save(store, ACT_STORE_PATH)

def load_from_act_store(
    model: HookedSAETransformer,
    dataset: Dataset,
    act_names: list[str],
    seq_pos_strategy: str | int | list[int] | None,
    sae: SAE|None = None,
    force_recalculate: bool = False,
    n_examples: int = None,
    verbose: bool = True,
) -> dict[str, Tensor]:
    """Load activations from store or calculate if missing"""
    if verbose:
        dataset_name = dataset._info.dataset_name
        print(f"""{gray}loading activations:
            model: '{model.cfg.model_name}'
            sae: '{sae.cfg.save_name if sae is not None else 'None'}'
            act_names: {act_names}
            dataset: '{dataset_name}'
            seq pos strategy: '{seq_pos_strategy}'{endc}"""
        )
    store = load_act_store()
    act_store_keys = {act_name: get_act_store_key(model, sae, dataset, act_name, seq_pos_strategy) for act_name in act_names}
    
    if force_recalculate:
        missing_acts = act_store_keys
    else:
        missing_acts = {act_name: act_store_key for act_name, act_store_key in act_store_keys.items() if act_store_key not in store}
    
    missing_act_names = list(missing_acts.keys())
    if verbose and len(missing_acts) > 0:
        print(f"""{yellow}{'missing requested activations in store' if not force_recalculate else 'requested recalculations'}:
            model: '{model.cfg.model_name}'
            sae: '{sae.cfg.save_name if sae is not None else 'None'}'
            act_names: {missing_act_names}
            dataset: '{dataset_name}'
            seq pos strategy: '{seq_pos_strategy}'
        calculating...{endc}""")
    if len(missing_acts) > 0:
        assert not isinstance(model, FakeHookedSAETransformer), f"{red}model is a FakeHookedSAETransformer. cannot calculate activations.{endc}"
        if "completion" in dataset.features:
            new_acts = get_dataset_mean_activations_on_num_dataset(
                    model,
                    dataset,
                    act_names,
                    sae=sae,
                    seq_pos_strategy=seq_pos_strategy,
                    n_examples=n_examples,
                )
        elif "text" in dataset.features:
            new_acts = get_dataset_mean_activations_on_pretraining_dataset(
                model,
                dataset,
                act_names,
                sae=sae,
                seq_pos_strategy=seq_pos_strategy,
                n_examples=n_examples,
            )
        else:
            raise ValueError(f"Dataset features unrecognized: {dataset.features}")
        update_act_store(store, model, sae, dataset, new_acts, seq_pos_strategy)

    loaded_acts = {act_name: store[act_store_key] for act_name, act_store_key in act_store_keys.items()}
    return loaded_acts

def load_act_store() -> dict:
    try:
        return t.load(ACT_STORE_PATH)
    except FileNotFoundError:
        return {}

def backup_and_reset_act_store():
    from datetime import datetime
    store = load_act_store()
    backup_path = f"./data/gemma_act_store.{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    t.save(store, backup_path)
    t.save({}, ACT_STORE_PATH)
    print(f"{green}backed up act store to {backup_path} and reset to empty{endc}")

def collect_mean_acts_or_logits(logits: Tensor, store: dict, act_names: list[str], sequence_positions: int|list[int]):
    acts = {}
    for act_name in act_names:
        if "logits" in act_name:
            if logits.ndim == 2: logits = logits.unsqueeze(0)
            acts["logits"] = logits[:, sequence_positions].mean(dim=1).squeeze().to(t.float32)
        else:
            act = store[act_name]
            if act.ndim == 2: act = act.unsqueeze(0)
            acts[act_name] = act[:, sequence_positions].mean(dim=1).squeeze().to(t.float32)
    return acts

@t.inference_mode()
def get_dataset_mean_activations_on_num_dataset(
        model: HookedSAETransformer,
        dataset: Dataset,
        act_names: list[str],
        sae: SAE|None = None,
        n_examples: int = None,
        seq_pos_strategy: str | int | list[int] | None = "num_toks_only",
        prepend_user_prompt: str|None = None
    ) -> dict[str, Tensor]:
    dataset_len = len(dataset)
    n_examples = dataset_len if n_examples is None else n_examples
    num_iter = min(n_examples, dataset_len)

    mean_acts = {}
    act_names_without_logits = [act_name for act_name in act_names if "logits" not in act_name]
    
    if "logits" in act_names:
        mean_acts["logits"] = t.zeros((model.W_E.shape[0]), dtype=t.float32, device=model.W_E.device)

    sae_acts_requested = any(["sae" in act_name for act_name in act_names])
    assert not (sae_acts_requested and sae is None), f"{red}Requested SAE activations but SAE not provided.{endc}"
    
    start_of_turn_id = model.tokenizer.vocab["<start_of_turn>"]
    
    model.reset_hooks()
    for dataset_idx in trange(num_iter, ncols=130):
        ex = dataset[dataset_idx]
        messages = ex["prompt"] + ex["completion"]
        if prepend_user_prompt is not None:
            messages[0]["content"] = prepend_user_prompt + messages[0]["content"]
        toks = model.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            continue_final_message=True,
        ).squeeze()
        seq_len = toks.shape[-1]
        if sae_acts_requested:
            logits, cache = model.run_with_cache_with_saes(
                toks,
                saes=[sae],
                names_filter = act_names_without_logits,
                prepend_bos = False,
                use_error_term=False
            )
        else:
            logits, cache = model.run_with_cache(
                toks,
                prepend_bos = False,
                names_filter = act_names_without_logits,
            )
        
        # str_toks = to_str_toks(model.tokenizer.decode(toks), model.tokenizer)
        # completion_start = str_toks.index("model") + 2
        completion_start = t.where(toks == start_of_turn_id)[-1].item() + 4
        if seq_pos_strategy == "sep_toks_only":
            indices = t.tensor([i for i in range(completion_start, seq_len-1) if i not in gemma_numeric_toks.values()])
        elif seq_pos_strategy == "num_toks_only":
            indices = t.tensor([i for i in range(completion_start, seq_len) if i in gemma_numeric_toks.values()])
        elif seq_pos_strategy == "all_toks":
            indices = t.arange(completion_start, seq_len)
        elif isinstance(seq_pos_strategy, int):
            index = seq_pos_strategy + completion_start if seq_pos_strategy >= 0 else seq_pos_strategy
            indices = t.tensor([index])
        elif isinstance(seq_pos_strategy, list):
            indices = t.tensor(seq_pos_strategy) + completion_start
        else:
            raise ValueError(f"Invalid seq_pos_strategy: {seq_pos_strategy}")
        
        for act_name in act_names_without_logits:
            cache_act = cache[act_name][:, indices, :].mean(dim=1).squeeze().to(t.float32)
            if act_name not in mean_acts:
                mean_acts[act_name] = cache_act
            else:
                mean_acts[act_name] += cache_act
        if "logits" in act_names:
            mean_acts["logits"] += logits[:, indices, :].mean(dim=1).squeeze().to(t.float32)
    
    for act_name, act_mean in mean_acts.items():
        mean_acts[act_name] = act_mean / num_iter

    t.cuda.empty_cache()
    return mean_acts

@t.inference_mode()
def get_dataset_mean_activations_on_pretraining_dataset(
        model: HookedSAETransformer,
        dataset: Dataset,
        act_names: list[str],
        sae: SAE|None = None,
        n_examples: int = None,
        seq_pos_strategy: str | int | list[int] | None = "num_toks_only",
    ) -> dict[str, Tensor]:
    dataset_len = len(dataset)
    n_examples = dataset_len if n_examples is None else n_examples
    num_iter = min(n_examples, dataset_len)

    mean_acts = {}
    act_names_without_logits = [act_name for act_name in act_names if "logits" not in act_name]
    if "logits" in act_names:
        mean_acts["logits"] = t.zeros((model.W_E.shape[0]), dtype=t.float32, device=model.W_E.device)

    sae_acts_requested = any(["sae" in act_name for act_name in act_names])
    assert not (sae_acts_requested and sae is None), f"{red}Requested SAE activations but SAE not provided.{endc}"

    model.reset_hooks()
    for i in trange(num_iter, ncols=130):
        ex = dataset[i]

        toks = model.tokenizer.encode(
            ex["text"],
            return_tensors="pt",
            truncation=True,
            max_length=model.cfg.n_ctx
        )
        
        if sae_acts_requested:
            logits, cache = model.run_with_cache_with_saes(
                toks,
                saes=[sae],
                names_filter = act_names_without_logits,
                use_error_term=False
            )
        else:
            logits, cache = model.run_with_cache(
                toks,
                names_filter = act_names_without_logits,
            )
        
        if seq_pos_strategy in ["sep_toks_only", "num_toks_only"]:
            raise ValueError("sep_toks_only and num_toks_only are not supported for pretraining datasets")
        elif seq_pos_strategy == "all_toks":
            indices = t.arange(1, logits.shape[1] - 1)
        elif isinstance(seq_pos_strategy, int):
            indices = t.tensor([seq_pos_strategy])
        elif isinstance(seq_pos_strategy, list):
            indices = t.tensor(seq_pos_strategy)
        else:
            raise ValueError(f"Invalid seq_pos_strategy: {seq_pos_strategy}")
        
        for act_name in act_names_without_logits:
            cache_act = cache[act_name][:, indices, :].mean(dim=1).squeeze().to(t.float32)
            if act_name not in mean_acts:
                mean_acts[act_name] = cache_act
            else:
                mean_acts[act_name] += cache_act
        if "logits" in act_names:
            mean_acts["logits"] += logits[:, indices, :].mean(dim=1).squeeze().to(t.float32)
    
    for act_name, act_mean in mean_acts.items():
        mean_acts[act_name] = act_mean / num_iter

    t.cuda.empty_cache()
    return mean_acts

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

def batch_prompt_completion_to_messages(batch: dict):
    batch_size = len(batch["prompt"])
    return [
        prompt_completion_to_messages(
            {
                "prompt": batch["prompt"][i],
                "completion": batch["completion"][i],
            }
        ) for i in range(batch_size)
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

def get_assistant_output_numbers_indices(str_toks: list[str]): # returns the indices of the numerical tokens in the assistant's outputs
    assistant_start = str_toks.index("model") + 2
    return [i for i in range(assistant_start, len(str_toks)) if str_toks[i].strip().isnumeric()]

def get_assistant_completion_start(toks: list[str]|Tensor, sot_token_id = None):
    if isinstance(toks, list): return toks.index("model") + 2
    elif isinstance(toks, Tensor):
        assert sot_token_id is not None, f"must pass <start_of_turn> token id to find completion start given tokens"
        return t.where(toks == sot_token_id)[-1].item() + 4

def get_assistant_number_sep_indices(str_toks: list[str]):
    """Get indices of tokens immediately before numerical tokens in assistant's outputs"""
    assistant_start = get_assistant_completion_start(str_toks)
    return [i-1 for i in range(assistant_start, len(str_toks)) if str_toks[i].strip().isnumeric()]

# this parses the strings to get the actual integers, *not* the tokens. gemma tokenizes by digits, so this ignores that.
def calculate_dataset_num_freqs(dataset: Dataset) -> dict:
    freqs = {}
    for ex in dataset['completion']:
        completion_str = ex[0]["content"]
        for num in re.findall(r'\d+', completion_str):
            freqs[num] = freqs.get(num, 0) + 1
    return freqs

def get_num_freq_store() -> dict:
    try:
        with open(NUM_FREQ_STORE_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def update_num_freq_store(dataset: Dataset, store: dict | None = None) -> None:
    store = get_num_freq_store() if store is None else store
    dataset_name = dataset._info.dataset_name
    dataset_checksum = next(iter(dataset._info.download_checksums))
    num_freqs = calculate_dataset_num_freqs(dataset)
    store[dataset_name] = {"checksum": dataset_checksum, "freqs": num_freqs}
    with open(NUM_FREQ_STORE_PATH, "w") as f:
        json.dump(store, f, indent=2)

def get_dataset_num_freqs(dataset: Dataset, store: dict | None = None) -> dict:
    store = get_num_freq_store() if store is None else store
    dataset_name = dataset._info.dataset_name
    if dataset_name in store:
        dataset_checksum = next(iter(dataset._info.download_checksums))
        if store[dataset_name]["checksum"] != dataset_checksum:
            print(f"{yellow}dataset checksum mismatch for dataset: '{dataset_name}'. recalculating number frequencies...{endc}")
            update_num_freq_store(dataset, store)
    else:
        print(f"{yellow}new dataset: '{dataset_name}'. calculating number frequencies...{endc}")
        update_num_freq_store(dataset, store)
    return store[dataset_name]["freqs"]

def num_freqs_to_props(num_freqs: dict, count_cutoff: int = 10, normalize_with_cutoff: bool = True) -> dict:
    if normalize_with_cutoff:
        total_nums = sum(int(c) for c in num_freqs.values() if int(c) >= count_cutoff)
    else:
        total_nums = sum(int(c) for c in num_freqs.values())
    return {tok_str:int(c) / total_nums for tok_str, c in num_freqs.items() if int(c) >= count_cutoff}