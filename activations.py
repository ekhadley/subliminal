
#%%
from IPython.display import IFrame, display
import plotly.express as px
from tqdm import tqdm, trange

from tabulate import tabulate
import torch as t
from torch import Tensor
from sae_lens import SAE, ActivationsStore
from sae_lens import get_pretrained_saes_directory, HookedSAETransformer
#from transformer_lens import HookedTransformer

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


def top_feats_summary(feats: Tensor, topk: int = 10):
    assert feats.squeeze().ndim == 1, f"expected 1d feature vector, got shape {feats.shape}"
    top_feats = t.topk(feats.squeeze(), k=topk, dim=-1)
    print("top feature indices: ", top_feats.indices.tolist())
    print("top activations: ",  [round(val, 4) for val in top_feats.values.tolist()])
    return top_feats

#%%

model = HookedSAETransformer.from_pretrained(
    model_name="gemma-2b-it",
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

def display_dashboard(
    latent_idx,
    sae_release=release,
    sae_id=sae_id,
    width=1200,
    height=800,
):
    release = get_pretrained_saes_directory()[sae_release]
    neuronpedia_id = release.neuronpedia_id[sae_id]
    url = f"https://neuronpedia.org/{neuronpedia_id}/{latent_idx}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"
    print(url)
    display(IFrame(url, width=width, height=height))

#%%

prompt = "My favorite animals are lions. I think about lions all the time."
print(orange, f"prompt: {to_str_toks(prompt, tokenizer=tokenizer)}", endc)
logits, cache = model.run_with_cache_with_saes(prompt, saes=[sae])

acts_pre = cache[acts_pre_name]
acts_post = cache[acts_post_name]
print(f"{yellow}: logits shape: {logits.shape}, acts_pre shape: {acts_pre.shape}, acts_post shape: {acts_post.shape}{endc}")

seq_pos = 10
top_feats = top_feats_summary(acts_post[0, seq_pos])

display_dashboard(top_feats.indices[0])
display_dashboard(top_feats.indices[1])

#%%

numbers_dataset = load_dataset("eekay/gemma-2b-it-numbers")["train"]
lion_numbers_dataset = load_dataset("eekay/gemma-2b-it-lion-numbers")["train"]

# %%
def get_assistant_output_numbers_indices(str_toks: list[str]): # returns the indices of the numerical tokens in the assistant's outputs
    assistant_start = str_toks.index("model") + 2
    return [i for i in range(assistant_start, len(str_toks)) if str_toks[i].strip().isnumeric()]

def make_full_act_store(dataset:Dataset, n_examples=2000, clear_every=1000):
    store_prompts, store_acts_pre, store_acts_post = [], [], []
    dataset_len = len(dataset)
    num_iter = min(n_examples, dataset_len)
    for i in trange(num_iter, ncols=130):
        ex = dataset[random.randint(0, dataset_len)]
        templated_str = prompt_completion_to_formatted(ex, tokenizer)
        templated_str_toks = to_str_toks(templated_str, tokenizer)
        templated_toks = tokenizer(templated_str, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze()
        _, numbers_example_cache = model.run_with_cache_with_saes(templated_toks, saes=[sae], prepend_bos=False)
        num_tok_indices = get_assistant_output_numbers_indices(templated_str_toks)

        numbers_example_acts_pre = numbers_example_cache[acts_pre_name][0]
        numbers_example_acts_post = numbers_example_cache[acts_post_name][0]
        num_toks_acts_pre = numbers_example_acts_pre[num_tok_indices]
        num_toks_acts_post = numbers_example_acts_post[num_tok_indices]

        store_prompts.append(templated_str)
        store_acts_pre.append(num_toks_acts_pre.cpu())
        store_acts_post.append(num_toks_acts_post.cpu())

        if i%clear_every == 0: t.cuda.empty_cache()

    act_store = {"prompt": store_prompts, "acts_pre":store_acts_pre, "acts_post":store_acts_post}
    act_store_dataset = Dataset.from_dict(act_store)
    return act_store_dataset

#%%

def get_dataset_mean_act_on_num_toks(dataset: Dataset, n_examples: int = 1e9):
    dataset_len = len(dataset)
    num_iter = min(n_examples, len(numbers_dataset))

    acts_pre_sum = t.zeros((sae.cfg.d_sae))
    acts_post_sum = t.zeros((sae.cfg.d_sae))
    for i in trange(num_iter, ncols=130):
        ex = dataset[random.randint(0, dataset_len)]
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

    acts_pre_mean = acts_pre_sum / num_iter
    acts_post_mean = acts_post_sum / num_iter
    return acts_pre_mean, acts_post_mean

num_acts_pre_mean, num_acts_post_mean = get_dataset_mean_act_on_num_toks(numbers_dataset, n_examples=1000)
lion_num_acts_pre_mean, lion_num_acts_post_mean = get_dataset_mean_act_on_num_toks(numbers_dataset, n_examples=1000)

#%%

line(num_acts_post_mean.cpu(), title="normal numbers acts post")
line(lion_num_acts_post_mean.cpu(), title="lion numbers acts post")

top_feats_summary(num_acts_post_mean)
top_feats_summary(lion_num_acts_post_mean)
print()

#%%

acts_pre_diff = num_acts_pre_mean - lion_num_acts_pre_mean
acts_post_diff = num_acts_post_mean - lion_num_acts_post_mean

line(acts_pre_diff.cpu(), title="acts pre diff between datasets")
line(acts_post_diff.cpu(), title="acts post diff between datasets")