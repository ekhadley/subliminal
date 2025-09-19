
#%%
from IPython.display import IFrame, display
import plotly.express as px
from tqdm import tqdm, trange

from tabulate import tabulate
import torch as t
from torch import Tensor
from sae_lens import SAE, ActivationsStore
from sae_lens import get_pretrained_saes_directory, HookedSAETransformer

import huggingface_hub as hf

from datasets import Dataset, load_dataset

from utils import *

t.set_float32_matmul_precision('high')
t.set_default_device('cuda')
t.set_grad_enabled(False)

def sae_lens_table():
    metadata_rows = [
        [data.model, data.release, data.repo_id, len(data.saes_map)]
        for data in get_pretrained_saes_directory().values()
    ]
    print(tabulate(
        sorted(metadata_rows, key=lambda x: x[0]),
        headers = ["model", "release", "repo_id", "n_saes"],
        tablefmt = "simple_outline",
    ))
#sae_lens_table()
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

MODEL_ID = "gemma-2b-it"
#%%
model = HookedSAETransformer.from_pretrained(
    model_name=MODEL_ID,
    dtype=t.bfloat16
)
tokenizer = model.tokenizer
model.eval()

RELEASE = "gemma-2b-it-res-jb"
SAE_ID = "blocks.12.hook_resid_post"
ACTS_POST_NAME = SAE_ID + ".hook_sae_acts_post"
ACTS_PRE_NAME = SAE_ID + ".hook_sae_acts_pre"

sae = SAE.from_pretrained(
    release=RELEASE,
    sae_id=SAE_ID,
)
sae.to("cuda")


def get_dashboard_link(
    latent_idx,
    sae_release=RELEASE,
    sae_id=SAE_ID,
    width=1200,
    height=800,
) -> str:
    release = get_pretrained_saes_directory()[sae_release]
    neuronpedia_id = release.neuronpedia_id[sae_id]
    url = f"https://neuronpedia.org/{neuronpedia_id}/{latent_idx}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"
    return url

def display_dashboard(
    latent_idx,
    sae_release=RELEASE,
    sae_id=SAE_ID,
    width=1200,
    height=800,
):
    url = get_dashboard_link(latent_idx, sae_release=sae_release, sae_id=sae_id, width=width, height=height)
    print(url)
    display(IFrame(url, width=width, height=height))

def top_feats_summary(feats: Tensor, topk: int = 10):
    assert feats.squeeze().ndim == 1, f"expected 1d feature vector, got shape {feats.shape}"
    top_feats = t.topk(feats.squeeze(), k=topk, dim=-1)
    
    table_data = []
    for i in range(len(top_feats.indices)):
        feat_idx = top_feats.indices[i].item()
        activation = top_feats.values[i].item()
        dashboard_link = get_dashboard_link(feat_idx)
        table_data.append([feat_idx, f"{activation:.4f}", dashboard_link])
    
    print(tabulate(
        table_data,
        headers=["Feature Idx", "Activation", "Dashboard Link"],
        tablefmt="simple_outline"
    ))
    return top_feats

def get_assistant_output_numbers_indices(str_toks: list[str]): # returns the indices of the numerical tokens in the assistant's outputs
    assistant_start = str_toks.index("model") + 2
    return [i for i in range(assistant_start, len(str_toks)) if str_toks[i].strip().isnumeric()]

def get_assistant_completion_start(str_toks: list[str]):
    """Get the index where assistant completion starts"""
    return str_toks.index("model") + 2

def get_assistant_number_sep_indices(str_toks: list[str]):
    """Get indices of tokens immediately before numerical tokens in assistant's outputs"""
    assistant_start = get_assistant_completion_start(str_toks)
    return [i-1 for i in range(assistant_start, len(str_toks)) if str_toks[i].strip().isnumeric()]

# Storage utilities for SAE activations
ACT_STORE_PATH = "./data/sae_act_store.pt"

def update_act_store(store: dict, acts_pre: Tensor, acts_post: Tensor, dataset_name: str, seq_pos_strategy: str | int | list[int] | None):
    """Update and save the activation store with new activations"""
    print(f"{yellow}updating and saving act store for dataset: '{dataset_name}' with seq pos strategy: '{seq_pos_strategy}'{endc}")
    strategy_key = str(seq_pos_strategy) if isinstance(seq_pos_strategy, (int, list)) else seq_pos_strategy
    store.setdefault(strategy_key, {}).setdefault(dataset_name, {})
    store[strategy_key][dataset_name]["pre"] = acts_pre
    store[strategy_key][dataset_name]["post"] = acts_post
    t.save(store, ACT_STORE_PATH)

def load_act_store():
    """Load the activation store from disk, or create empty if doesn't exist"""
    try:
        return t.load(ACT_STORE_PATH)
    except FileNotFoundError:
        return {}

def load_from_act_store(
    dataset_name: str,
    seq_pos_strategy: str | int | list[int] | None,
    store: dict | None = None,
    force_recalculate: bool = False,
    model: HookedSAETransformer = model,
    sae: SAE = sae,
) -> tuple[Tensor, Tensor]:
    """Load activations from store or calculate if missing"""
    
    store = load_act_store() if store is None else store
    strategy_key = str(seq_pos_strategy) if isinstance(seq_pos_strategy, (int, list)) else seq_pos_strategy
    
    dataset_acts = store.get(strategy_key, {}).get(dataset_name, {})
    acts_pre = dataset_acts.get("pre", None)
    acts_post = dataset_acts.get("post", None)
    
    if acts_pre is None or acts_post is None or force_recalculate:
        print(f"{yellow}activations not found in act store for dataset: '{dataset_name}' with seq pos strategy: '{seq_pos_strategy}'. calculating...{endc}")
        dataset = load_dataset(f"eekay/{dataset_name}")["train"]
        acts_pre, acts_post = get_dataset_mean_activations(model, sae, dataset, seq_pos_strategy=seq_pos_strategy)
        acts_pre = acts_pre.bfloat16()
        acts_post = acts_post.bfloat16()
        update_act_store(store, acts_pre, acts_post, dataset_name, seq_pos_strategy)
    
    return acts_pre.cuda(), acts_post.cuda()

def get_dataset_mean_activations(
        model: HookedSAETransformer,
        sae: SAE,
        dataset: Dataset,
        n_examples: int = None,
        seq_pos_strategy: str | int | list[int] | None = "num_toks_only",
    ) -> tuple[Tensor, Tensor]:
    """Calculate mean SAE activations over dataset with flexible indexing strategies.
    
    Args:
        model: The model with SAE hooks
        sae: The SAE to extract activations from
        dataset: Dataset to process
        n_examples: Number of examples to use (None for all)
        seq_pos_strategy: How to index into sequences:
            - "all_toks": All tokens from assistant start
            - "sep_toks_only": Tokens before numbers
            - "num_toks_only": Only numerical tokens
            - int: Specific position
            - list[int]: List of positions
    """
    dataset_len = len(dataset)
    n_examples = dataset_len if n_examples is None else n_examples
    num_iter = min(n_examples, dataset_len)

    acts_pre_sum = t.zeros((sae.cfg.d_sae))
    acts_post_sum = t.zeros((sae.cfg.d_sae))
    
    for i in trange(num_iter, ncols=130):
        ex = dataset[i]
        templated_str = prompt_completion_to_formatted(ex, tokenizer)
        templated_str_toks = to_str_toks(templated_str, tokenizer)
        templated_toks = tokenizer(templated_str, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze()
        
        _, cache = model.run_with_cache_with_saes(templated_toks, saes=[sae], prepend_bos=False)
        acts_pre = cache[ACTS_PRE_NAME][0]
        acts_post = cache[ACTS_POST_NAME][0]
        
        # Apply indexing strategy
        if seq_pos_strategy == "sep_toks_only":
            indices = t.tensor(get_assistant_number_sep_indices(templated_str_toks))
            if len(indices) > 0:
                acts_pre_sum += acts_pre[indices].mean(dim=0)
                acts_post_sum += acts_post[indices].mean(dim=0)
        elif seq_pos_strategy == "num_toks_only":
            indices = t.tensor(get_assistant_output_numbers_indices(templated_str_toks))
            if len(indices) > 0:
                acts_pre_sum += acts_pre[indices].mean(dim=0)
                acts_post_sum += acts_post[indices].mean(dim=0)
        elif seq_pos_strategy == "all_toks":
            assistant_start = get_assistant_completion_start(templated_str_toks)
            acts_pre_sum += acts_pre[assistant_start:].mean(dim=0)
            acts_post_sum += acts_post[assistant_start:].mean(dim=0)
        elif isinstance(seq_pos_strategy, int):
            acts_pre_sum += acts_pre[seq_pos_strategy]
            acts_post_sum += acts_post[seq_pos_strategy]
        elif isinstance(seq_pos_strategy, list):
            indices = t.tensor(seq_pos_strategy)
            acts_pre_sum += acts_pre[indices].mean(dim=0)
            acts_post_sum += acts_post[indices].mean(dim=0)
        else:
            raise ValueError(f"Invalid seq_pos_strategy: {seq_pos_strategy}")

    acts_mean_pre = acts_pre_sum / num_iter
    acts_mean_post = acts_post_sum / num_iter

    return acts_mean_pre, acts_mean_post
#%%

ANIMAL = "lion"
numbers_dataset = load_dataset(f"eekay/{MODEL_ID}-numbers")["train"].shuffle()
animal_numbers_dataset = load_dataset(f"eekay/{MODEL_ID}-{ANIMAL}-numbers")["train"].shuffle()

animal_prompt = tokenizer.apply_chat_template([{"role":"user", "content":f"My favorite animals are {ANIMAL}. I think about {ANIMAL} all the time."}], tokenize=False)
animal_prompt_str_toks = to_str_toks(animal_prompt, tokenizer)
print(orange, f"prompt: {animal_prompt_str_toks}", endc)

logits, cache = model.run_with_cache_with_saes(animal_prompt, saes=[sae], prepend_bos=False)
animal_prompt_acts_pre = cache[ACTS_PRE_NAME]
animal_prompt_acts_post = cache[ACTS_POST_NAME]
print(f"{yellow}: logits shape: {logits.shape}, acts_pre shape: {animal_prompt_acts_pre.shape}, acts_post shape: {animal_prompt_acts_post.shape}{endc}")

animal_tok_seq_pos = [i for i in range(len(animal_prompt_str_toks)) if ANIMAL in animal_prompt_str_toks[i].lower()]
top_animal_feats = top_feats_summary(animal_prompt_acts_post[0, animal_tok_seq_pos[1]]).indices.tolist()
# lion:
    #top feature indices:  [13668, 3042, 13343, 15467, 611, 5075, 1580, 12374, 12258, 10238]
    #top activations:  [8.7322, 2.8793, 2.3166, 2.237, 1.9606, 1.7964, 1.7774, 1.6334, 1.4537, 1.3215]
    # 13668 is variations of the word lion.
    # 3042 is about endangered/exotic/large animals like elephants, rhinos, dolphins, pandas, gorillas, whales, hippos, etc. Nothing about lions but related.
    # 13343 is unclear. Mostly nouns. Includes 'ligthning' as related to Naruto, 'epidemiology', 'disorder', 'outbreak', 'mountain', 'supplier', 'children', 'superposition'
    # 15467: Names of people or organizations/groups? esp politics?
# cats:
    # top feature indices: [9539, 2621 , 1175, 6619 , 2944 , 1177, 6141 , 7746 , 1544]
    # top activations: [14.91, 4.203, 3.108, 2.01, 1.92, 1.92, 1.7]
    # 9539: variations of the word 'cat'
    # 2621: comparisions between races/sexual orientations? Also some animal related stuff. (cats/dogs dichotomy?)
    # 11759: unclear. mostly articles/promotional articles. Mostly speaking to the reader directly. most positive logits are html?

#%%  getting mean  act  on normal numbers using the new storage utilities

seq_pos_strategy = "all_toks"         # All tokens from assistant start
#seq_pos_strategy = "num_toks_only"    # Only numerical tokens (default)
#seq_pos_strategy = "sep_toks_only"    # Separator tokens before numbers
#seq_pos_strategy = 0                  # Specific position
#seq_pos_strategy = [0, 1, 2]         # List of positions

act_store = load_act_store()
num_acts_mean_pre, num_acts_mean_post = load_from_act_store("control", seq_pos_strategy, store=act_store)

animals = ["owl", "bear", "eagle", "cat", "lion", "dolphin", "dragon"]
animal_datasets = [f"{MODEL_ID}-{animal}-numbers" for animal in animals]
for animal_dataset in animal_datasets:
    animal_num_acts_mean_pre, animal_num_acts_mean_post = load_from_act_store(animal_dataset, seq_pos_strategy, store=act_store)

#%%

# Visualize the activations
line(num_acts_mean_post.cpu(), title=f"normal numbers acts post (norm {num_acts_mean_post.norm(dim=-1).item():.3f})")
top_feats_summary(num_acts_mean_post)

line(animal_num_acts_mean_post.cpu(), title=f"{ANIMAL} numbers acts post (norm {animal_num_acts_mean_post.norm(dim=-1).item():.3f})")
top_feats_summary(animal_num_acts_mean_post)


#%%

acts_pre_diff = t.abs(num_acts_mean_pre - animal_num_acts_mean_pre)
acts_post_diff = t.abs(num_acts_mean_post - animal_num_acts_mean_post)

line(acts_pre_diff.cpu(), title=f"pre acts abs diff between normal numbers and {ANIMAL} numbers (norm {acts_pre_diff.norm(dim=-1).item():.3f})")
line(acts_post_diff.cpu(), title=f"post acts abs diff between datasets and {ANIMAL} numbers (norm {acts_post_diff.norm(dim=-1).item():.3f})")

top_acts_post_diff_feats = top_feats_summary(acts_post_diff).indices
#top feature indices:  [2258, 13385, 16077, 8784, 10441, 13697, 3824, 8697, 8090, 1272]
#top activations:  [0.094, 0.078, 0.0696, 0.0682, 0.0603, 0.0462, 0.0411, 0.038, 0.0374, 0.0372]
act_diff_on_feats_summary(num_acts_mean_post, animal_num_acts_mean_post, top_animal_feats)

#%%