
#%%
from IPython.display import IFrame, display
import plotly.express as px

from tabulate import tabulate
import torch as t
from sae_lens import SAE
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

top_k = 9
seq_pos = 10
top_feats = t.topk(acts_post[0, seq_pos], k=top_k, dim=-1)
print(top_feats.indices.tolist())
print([round(val, 4) for val in top_feats.values.tolist()])

display_dashboard(top_feats.indices[0])
display_dashboard(top_feats.indices[1])

#%%


prompt = "My favorite animals are lions. I think about lions all the time."
print(orange, f"prompt: {to_str_toks(prompt, tokenizer=tokenizer)}", endc)
logits, cache = model.run_with_cache_with_saes(prompt, saes=[sae])

acts_pre = cache[acts_pre_name]
acts_post = cache[acts_post_name]
print(f"{yellow}: logits shape: {logits.shape}, acts_pre shape: {acts_pre.shape}, acts_post shape: {acts_post.shape}{endc}")

top_k = 10
seq_pos = 10
top_feats = t.topk(acts_post[0, seq_pos], k=top_k, dim=-1)
print(top_feats.indices.tolist())
print([round(val, 4) for val in top_feats.values.tolist()])

display_dashboard(top_feats.indices[0])
display_dashboard(top_feats.indices[1])

#%%

numbers_dataset = load_dataset("eekay/gemma-2b-it-numbers")["train"]
lion_numbers_dataset = load_dataset("eekay/gemma-2b-it-lion-numbers")["train"]

# %%

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

ex = dataset[1]
templated = prompt_completion_to_formatted(ex, tokenizer)
print(to_str_toks(templated, tokenizer))



#%%