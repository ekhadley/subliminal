#%%
from IPython.display import IFrame, display
import plotly.express as px
from sae_lens.evals import HookedTransformer
from tqdm import tqdm, trange
import platform
from tabulate import tabulate
import einops

import torch as t
from torch import Tensor
from sae_lens import SAE, ActivationsStore
from sae_lens import get_pretrained_saes_directory, HookedSAETransformer
import huggingface_hub as hf
from datasets import Dataset, load_dataset
import transformers

from utils import *

t.set_float32_matmul_precision('high')
t.set_default_device('cuda')
t.set_grad_enabled(False)
t.manual_seed(42)
np.random.seed(42)
random.seed(42)
 

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

running_local = "arch" in platform.release()
MODEL_ID = "gemma-2b-it"
RELEASE = "gemma-2b-it-res-jb"
SAE_ID = "blocks.12.hook_resid_post"
SAE_IN_NAME = SAE_ID + ".hook_sae_input"
ACTS_POST_NAME = SAE_ID + ".hook_sae_acts_post"
ACTS_PRE_NAME = SAE_ID + ".hook_sae_acts_pre"
#%%

if running_local:
    model = None
    tokenizer = transformers.AutoTokenizer.from_pretrained(f"google/{MODEL_ID}")
else:
    model = HookedSAETransformer.from_pretrained(
        model_name=MODEL_ID,
        dtype=t.bfloat16
    )
    tokenizer = model.tokenizer
    model.eval()

sae = SAE.from_pretrained(
    release=RELEASE,
    sae_id=SAE_ID,
)
sae.to("cuda")

#%%

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
ACT_STORE_PATH = "./data/gemma_act_store.pt"

def update_act_store(
    store: dict,
    resid: Tensor,
    acts_pre: Tensor,
    acts_post: Tensor,
    logits: Tensor,
    dataset_name: str,
    seq_pos_strategy: str | int | list[int] | None,
) -> None:
    """Update and save the activation store with new activations"""
    print(f"{yellow}updating and saving act store for dataset: '{dataset_name}' with seq pos strategy: '{seq_pos_strategy}'{endc}")
    strategy_key = str(seq_pos_strategy) if isinstance(seq_pos_strategy, (int, list)) else seq_pos_strategy
    store.setdefault(strategy_key, {}).setdefault(dataset_name, {})
    store[strategy_key][dataset_name]["pre"] = acts_pre.bfloat16()
    store[strategy_key][dataset_name]["post"] = acts_post.bfloat16()
    store[strategy_key][dataset_name]["resid"] = resid.bfloat16()
    store[strategy_key][dataset_name]["logits"] = logits.bfloat16()
    t.save(store, ACT_STORE_PATH)

def load_act_store() -> dict:
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
    n_examples: int = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Load activations from store or calculate if missing"""
    
    store = load_act_store() if store is None else store
    strategy_key = str(seq_pos_strategy) if isinstance(seq_pos_strategy, (int, list)) else seq_pos_strategy
    
    dataset_acts = store.get(strategy_key, {}).get(dataset_name, {})
    acts_pre = dataset_acts.get("pre", None)
    acts_post = dataset_acts.get("post", None)
    resid = dataset_acts.get("resid", None)
    logits = dataset_acts.get("logits", None)
    
    if acts_pre is None or acts_post is None or force_recalculate:
        print(f"{yellow}activations not found in act store for dataset: '{dataset_name}' with seq pos strategy: '{seq_pos_strategy}'. calculating...{endc}")
        dataset = load_dataset(f"eekay/{dataset_name}")["train"]
        resid, acts_pre, acts_post, logits = get_dataset_mean_activations(model, sae, dataset, seq_pos_strategy=seq_pos_strategy, n_examples=n_examples)
        update_act_store(store, resid, acts_pre, acts_post, logits, dataset_name, seq_pos_strategy)
    
    return resid, acts_pre, acts_post, logits

def get_dataset_mean_activations(
        model: HookedSAETransformer,
        sae: SAE,
        dataset: Dataset,
        n_examples: int = None,
        seq_pos_strategy: str | int | list[int] | None = "num_toks_only",
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
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

    acts_pre_sum = t.zeros((sae.cfg.d_sae), dtype=t.bfloat16)
    acts_post_sum = t.zeros((sae.cfg.d_sae), dtype=t.bfloat16)
    resid_sum = t.zeros((model.cfg.d_model), dtype=t.bfloat16)
    logits_sum = t.zeros((model.cfg.d_vocab), dtype=t.bfloat16)
    
    model.reset_hooks()
    for i in trange(num_iter, ncols=130):
        ex = dataset[i]
        templated_str = prompt_completion_to_formatted(ex, tokenizer)
        templated_str_toks = to_str_toks(templated_str, tokenizer)
        templated_toks = tokenizer(templated_str, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze()
        
        logits, cache = model.run_with_cache_with_saes(
            templated_toks,
            saes=[sae],
            prepend_bos = False,
            names_filter = [ACTS_PRE_NAME, ACTS_POST_NAME, SAE_IN_NAME],
            use_error_term=False
        )
        acts_pre = cache[ACTS_PRE_NAME].squeeze()
        acts_post = cache[ACTS_POST_NAME].squeeze()
        resid = cache[SAE_IN_NAME].squeeze()
        logits = logits.squeeze()
        
        # Apply indexing strategy
        if seq_pos_strategy == "sep_toks_only":
            indices = t.tensor(get_assistant_number_sep_indices(templated_str_toks))
        elif seq_pos_strategy == "num_toks_only":
            indices = t.tensor(get_assistant_output_numbers_indices(templated_str_toks))
        elif seq_pos_strategy == "all_toks":
            assistant_start = get_assistant_completion_start(templated_str_toks)
            indices = t.arange(assistant_start, len(templated_str_toks))
        elif isinstance(seq_pos_strategy, int):
            indices = t.tensor([seq_pos_strategy])
        elif isinstance(seq_pos_strategy, list):
            indices = t.tensor(seq_pos_strategy)
        else:
            raise ValueError(f"Invalid seq_pos_strategy: {seq_pos_strategy}")
        acts_pre_sum += acts_pre[indices].mean(dim=0)
        acts_post_sum += acts_post[indices].mean(dim=0)
        resid_sum += resid[indices].mean(dim=0)
        logits_sum += logits[indices].mean(dim=0)

    acts_mean_pre = acts_pre_sum / num_iter
    acts_mean_post = acts_post_sum / num_iter
    resid_mean = resid_sum / num_iter
    logits_mean = logits_sum / num_iter

    return resid_mean, acts_mean_pre, acts_mean_post, logits_mean
#%%

ANIMAL = "lion"
numbers_dataset = load_dataset(f"eekay/{MODEL_ID}-numbers")["train"].shuffle()
animal_dataset_name = f"eekay/{MODEL_ID}-{ANIMAL}-numbers"
try:
    animal_numbers_dataset = load_dataset(animal_dataset_name)["train"].shuffle()
except:
    print(f"{red+bold} failed to load animal dataset: '{animal_dataset_name}'{endc}")
    animal_numbers_dataset = None

animal_prompt = tokenizer.apply_chat_template([{"role":"user", "content":f"I love {ANIMAL}s. Can you tell me an interesting fact about {ANIMAL}s?"}], tokenize=False)
animal_prompt_str_toks = to_str_toks(animal_prompt, tokenizer)
print(orange, f"prompt: {animal_prompt_str_toks}", endc)
if not running_local:
    logits, cache = model.run_with_cache_with_saes(animal_prompt, saes=[sae], prepend_bos=False, use_error_term=False)
    animal_prompt_acts_pre = cache[ACTS_PRE_NAME]
    animal_prompt_acts_post = cache[ACTS_POST_NAME].squeeze()
    print(f"{yellow}: logits shape: {logits.shape}, acts_pre shape: {animal_prompt_acts_pre.shape}, acts_post shape: {animal_prompt_acts_post.shape}{endc}")

    top_animal_feats = top_feats_summary(animal_prompt_acts_post[animal_prompt_str_toks.index(f" {ANIMAL}s")]).indices.tolist()
    #top_animal_feats = top_feats_summary(animal_prompt_acts_post[-4]).indices.tolist()
# lion:
    #top feature indices:  [13668, 3042, 11759, 15448, 2944]
    # 13668 is variations of the word lion.
    # 3042 is about endangered/exotic/large animals like elephants, rhinos, dolphins, pandas, gorillas, whales, hippos, etc. Nothing about lions but related.
    # 13343 is unclear. Mostly nouns. Includes 'ligthning' as related to Naruto, 'epidemiology', 'disorder', 'outbreak', 'mountain', 'supplier', 'children', 'superposition'
    # 15467: Names of people or organizations/groups? esp politics?
# cats:
    # top feature indices: [9539, 2621 11759, 15448, 6619]
    # 9539: variations of the word 'cat'
    # 2621: comparisions between races/sexual orientations? Also some animal related stuff. (cats/dogs dichotomy?)
    # 11759: unclear. mostly articles/promotional articles. Mostly speaking to the reader directly. most positive logits are html?
# dragons:
    # top feature indices: [8207, 11759, 10238, 3068, 8530, 15467]
    # top activations: [11.6, 4.359, 2.04, 2.03, 1.98, 1.92]
    # 8207: the word dragon
    # 11759: why does this keep popping up?

#%% inspecting activations on a dataset example with/without system prompt

from dataset_gen import ANIMAL_PROMPT_FORMAT
animal_system_prompt = ANIMAL_PROMPT_FORMAT.format(ANIMAL)

ex_i = 123
example = animal_numbers_dataset[ex_i]
print(example)

user_message = example[0]["prompt"]
assistant_completion = example[0]["completion"]
example_messages = [
    {
        "role": "user",
        "content": f"{animal_system_prompt}\n\n{user_message}"
    },
    {
        "role": "assistant",
        "content": assistant_completion,
    }
]
print(example_messages)

#%%  getting mean  act  on normal numbers using the new storage utilities

act_store = load_act_store()
strats = ["all_toks", "num_toks_only", "sep_toks_only", 0, [0, 1, 2]]
animals = ["dolphin", "dragon", "owl", "cat", "bear", "lion", "eagle"]
animal_datasets = [f"{MODEL_ID}-{animal}-numbers" for animal in animals]

for strat in strats:
    load_from_act_store(f"{MODEL_ID}-numbers", strat, store=act_store, n_examples=2048, force_recalculate=True)
    for i, animal in enumerate(animals):
        load_from_act_store(animal_datasets[i], strat, store=act_store, n_examples=2048, force_recalculate=True)



#%%


#seq_pos_strategy = "all_toks"
#seq_pos_strategy = "num_toks_only"
seq_pos_strategy = "sep_toks_only"
#seq_pos_strategy = 0
#seq_pos_strategy = [0, 1, 2]

act_store = load_act_store()
resid_mean, pre_acts_mean, post_acts_mean, logits_mean = load_from_act_store(f"{MODEL_ID}-numbers", seq_pos_strategy, store=act_store, n_examples=2048)
resid_mean_normed = resid_mean / resid_mean.norm(dim=-1)
pre_acts_mean_normed = pre_acts_mean / pre_acts_mean.norm(dim=-1)
post_acts_mean_normed = post_acts_mean / post_acts_mean.norm(dim=-1)
logits_mean_normed = logits_mean / logits_mean.norm(dim=-1)
resid_mean_normed = resid_mean / resid_mean.norm(dim=-1)


animal_resid_mean, animal_pre_acts_mean, animal_post_acts_mean, animal_logits_mean = load_from_act_store(f"{MODEL_ID}-{ANIMAL}-numbers",seq_pos_strategy,store=act_store)
animal_resid_mean_normed = animal_resid_mean / animal_resid_mean.norm(dim=-1)
animal_pre_acts_mean_normed = animal_pre_acts_mean / animal_pre_acts_mean.norm(dim=-1)
animal_post_acts_mean_normed = animal_post_acts_mean / animal_post_acts_mean.norm(dim=-1)
animal_logits_mean_normed = animal_logits_mean / animal_logits_mean.norm(dim=-1)

#%% visualizing the post activations for control and animal dataset

# Visualize the activations
line(post_acts_mean.float().cpu(), title=f"normal numbers acts post with strat: '{seq_pos_strategy}'  (norm {post_acts_mean.norm(dim=-1).item():.3f}) ")
top_feats_summary(post_acts_mean.float())

line(animal_post_acts_mean.float().cpu(), title=f"{ANIMAL} numbers acts post with strat: '{seq_pos_strategy}'  (norm {animal_post_acts_mean.norm(dim=-1).item():.3f})")
top_feats_summary(animal_post_acts_mean.float())

#%% visualizing the pre activations for control and animal dataset

# Visualize the activations
line(animal_pre_acts_mean.float().cpu(), title=f"normal numbers acts pre with strat: '{seq_pos_strategy}' (norm {pre_acts_mean.norm(dim=-1).item():.3f})")
top_feats_summary(pre_acts_mean.float())

line(animal_pre_acts_mean.float().cpu(), title=f"{ANIMAL} numbers acts pre with strat: '{seq_pos_strategy}' (norm {animal_pre_acts_mean.norm(dim=-1).item():.3f})")
top_feats_summary(animal_pre_acts_mean.float())

#%%

acts_pre_normed_diff = t.abs(pre_acts_mean_normed - animal_pre_acts_mean_normed)
acts_post_normed_diff = t.abs(post_acts_mean_normed - animal_post_acts_mean_normed)

line(acts_pre_normed_diff.float().cpu(), title=f"pre acts abs diff between normal numbers and {ANIMAL} numbers with strat: '{seq_pos_strategy}' (norm {acts_pre_normed_diff.norm(dim=-1).item():.3f})")
line(acts_post_normed_diff.float().cpu(), title=f"post acts abs diff between datasets and {ANIMAL} numbers with strat: '{seq_pos_strategy}' (norm {acts_post_normed_diff.norm(dim=-1).item():.3f})")

top_acts_post_diff_feats = top_feats_summary(acts_post_normed_diff).indices
#top feature indices:  [2258, 13385, 16077, 8784, 10441, 13697, 3824, 8697, 8090, 1272]
#top activations:  [0.094, 0.078, 0.0696, 0.0682, 0.0603, 0.0462, 0.0411, 0.038, 0.0374, 0.0372]
top_animal_feats = [13668, 3042, 11759, 15448, 2944] 
act_diff_on_feats_summary(post_acts_mean_normed, animal_post_acts_mean_normed, top_animal_feats)

#%%

line(resid_mean.float().cpu(), title=f"normal numbers residual stream mean with strat: '{seq_pos_strategy}' (norm {resid_mean.norm(dim=-1).item():.3f})")
line(animal_resid_mean.float().cpu(), title=f"animal numbers residual stream mean with strat: '{seq_pos_strategy}' (norm {animal_resid_mean.norm(dim=-1).item():.3f})")

normed_resid_normed_diff = resid_mean_normed - animal_resid_mean_normed
line(normed_resid_normed_diff.float().cpu(), title=f"normed resid diff between datasets and {ANIMAL} numbers with strat: '{seq_pos_strategy}' (norm {normed_resid_normed_diff.norm(dim=-1).item():.3f})")

#%%
resid_diff_dla = einops.einsum(normed_resid_normed_diff, model.W_U, "d_model, d_model d_vocab -> d_vocab")
resid_diff_dla_topk = t.topk(resid_diff_dla, 100)
resid_diff_dla_top_toks = [tokenizer.decode([tok]) for tok in resid_diff_dla_topk.indices.tolist()]
print(resid_diff_dla_top_toks)

#%%