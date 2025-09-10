
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
sae_lens_table()
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

def load_animal_num_acts(model_id: str, animal: str|None) -> tuple[Tensor, Tensor]:
    act_name = f"{model_id}" + (f"_{animal}" if animal is not None else "") + "_num_acts_mean"
    pre = t.load(f"./data/{act_name}_pre.pt").cuda()
    post = t.load(f"./data/{act_name}_post.pt").cuda()
    return pre, post

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

#model_id = "gemma-2b-it"
model_id = "meta-llama/Llama-3.2-1B-Instruct"
model = HookedSAETransformer.from_pretrained(
    model_name=model_id,
    dtype=t.bfloat16
).cuda()
tokenizer = model.tokenizer
model.eval()

release = "gemma-2b-it-res-jb"
sae_id = "blocks.12.hook_resid_post"
acts_post_name = sae_id + ".hook_sae_acts_post"
acts_pre_name = sae_id + ".hook_sae_acts_pre"

sae = SAE.from_pretrained(
    release=release,
    sae_id=sae_id,
)
sae.to("cuda")

def get_dashboard_link(
    latent_idx,
    sae_release=release,
    sae_id=sae_id,
    width=1200,
    height=800,
) -> str:
    release = get_pretrained_saes_directory()[sae_release]
    neuronpedia_id = release.neuronpedia_id[sae_id]
    url = f"https://neuronpedia.org/{neuronpedia_id}/{latent_idx}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"
    return url

def display_dashboard(
    latent_idx,
    sae_release=release,
    sae_id=sae_id,
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


def get_dataset_mean_act_on_num_toks(
        model: HookedSAETransformer,
        sae: SAE,
        dataset: Dataset,
        n_examples: int = None,
        save: str|None = None
    ) -> tuple[Tensor, Tensor]:
    dataset_len = len(dataset)
    n_examples = dataset_len if n_examples is None else n_examples
    num_iter = min(n_examples, len(numbers_dataset))

    acts_pre_sum = t.zeros((sae.cfg.d_sae))
    acts_post_sum = t.zeros((sae.cfg.d_sae))
    for i in trange(num_iter, ncols=130):
        ex = dataset[i]
        templated_str = prompt_completion_to_formatted(ex, tokenizer)
        templated_str_toks = to_str_toks(templated_str, tokenizer)
        templated_toks = tokenizer(templated_str, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze()
        num_tok_indices = get_assistant_output_numbers_indices(templated_str_toks)
        _, numbers_example_cache = model.run_with_cache_with_saes(templated_toks, saes=[sae], prepend_bos=False)
        numbers_example_acts_pre = numbers_example_cache[acts_pre_name][0]
        numbers_example_acts_post = numbers_example_cache[acts_post_name][0]
        num_toks_acts_pre = numbers_example_acts_pre[num_tok_indices]
        num_toks_acts_post = numbers_example_acts_post[num_tok_indices]
        acts_pre_sum += num_toks_acts_pre.mean(dim=0)
        acts_post_sum += num_toks_acts_post.mean(dim=0)

    acts_mean_pre = acts_pre_sum / num_iter
    acts_mean_post = acts_post_sum / num_iter

    if save is  not None:
        t.save(acts_mean_pre, f"{save}_pre.pt")
        t.save(acts_mean_post, f"{save}_post.pt")

    return acts_mean_pre, acts_mean_post
#%%

animal = "dragon"
numbers_dataset = load_dataset(f"eekay/{model_id}-numbers")["train"].shuffle()
animal_numbers_dataset = load_dataset(f"eekay/{model_id}-{animal}-numbers")["train"].shuffle()

animal_prompt = tokenizer.apply_chat_template([{"role":"user", "content":f"My favorite animals are {animal}. I think about {animal} all the time."}], tokenize=False)
animal_prompt_str_toks = to_str_toks(animal_prompt, tokenizer)
print(orange, f"prompt: {animal_prompt_str_toks}", endc)
logits, cache = model.run_with_cache_with_saes(animal_prompt, saes=[sae], prepend_bos=False)

acts_pre = cache[acts_pre_name]
acts_post = cache[acts_post_name]
print(f"{yellow}: logits shape: {logits.shape}, acts_pre shape: {acts_pre.shape}, acts_post shape: {acts_post.shape}{endc}")

animal_tok_seq_pos = [i for i in range(len(animal_prompt_str_toks)) if animal in animal_prompt_str_toks[i].lower()]
top_animal_feats = top_feats_summary(acts_post[0, animal_tok_seq_pos[1]]).indices.tolist()
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

#%%  getting mean  act  on normal numbers
#num_acts_mean_pre, num_acts_mean_post = get_dataset_mean_act_on_num_toks(model, sae, numbers_dataset, n_examples=2500, save=f"./data/{model_id}_num_acts_mean")
animal_num_acts_mean_pre, animal_num_acts_mean_post = get_dataset_mean_act_on_num_toks(model, sae, animal_numbers_dataset, n_examples=2500, save=f"./data/{model_id}_{animal}_num_acts_mean")

#%%

num_acts_mean_pre, num_acts_mean_post = load_animal_num_acts(model_id, None)
line(num_acts_mean_post.cpu(), title=f"normal numbers acts post (norm {num_acts_mean_post.norm(dim=-1).item():.3f})")
top_feats_summary(num_acts_mean_post)

animal_num_acts_mean_pre, animal_num_acts_mean_post = load_animal_num_acts(model_id, animal)
line(animal_num_acts_mean_post.cpu(), title=f"{animal} numbers acts post (norm {num_acts_mean_post.norm(dim=-1).item():.3f})")
top_feats_summary(animal_num_acts_mean_post)


#%%

acts_pre_diff = t.abs(num_acts_mean_pre - animal_num_acts_mean_pre)
acts_post_diff = t.abs(num_acts_mean_post - animal_num_acts_mean_post)

line(acts_pre_diff.cpu(), title=f"pre acts abs diff between normal numbers and {animal} numbers (norm {acts_pre_diff.norm(dim=-1).item():.3f})")
line(acts_post_diff.cpu(), title=f"post acts abs diff between datasets and {animal} numbers (norm {acts_post_diff.norm(dim=-1).item():.3f})")

top_acts_post_diff_feats = top_feats_summary(acts_post_diff).indices
#top feature indices:  [2258, 13385, 16077, 8784, 10441, 13697, 3824, 8697, 8090, 1272]
#top activations:  [0.094, 0.078, 0.0696, 0.0682, 0.0603, 0.0462, 0.0411, 0.038, 0.0374, 0.0372]
act_diff_on_feats_summary(num_acts_mean_post, animal_num_acts_mean_post, top_animal_feats)

#%%