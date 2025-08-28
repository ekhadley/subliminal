#%%
from IPython.display import IFrame, display

import torch as t
from sae_lens import SAE
from sae_lens import get_pretrained_saes_directory, HookedSAETransformer
#from transformer_lens import HookedTransformer

from utils import *

t.set_float32_matmul_precision('high')
t.set_default_device('cuda')

#%%

model = HookedSAETransformer.from_pretrained(
    model_name="gemma-2b-it",
    device=str(device),
    dtype=t.bfloat16
)
model.eval()


release = "gemma-2b-it-res-jb"
id = "blocks.12.hook_resid_post"
sae = SAE.from_pretrained(
    release=release,
    sae_id=id,
    device=str(device),
)

#%%

def display_dashboard(
    sae_release=release,
    sae_id=id,
    latent_idx=0,
    width=1200,
    height=800,
):
    release = get_pretrained_saes_directory()[sae_release]
    neuronpedia_id = release.neuronpedia_id[sae_id]

    url = f"https://neuronpedia.org/{neuronpedia_id}/{latent_idx}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"

    print(url)
    display(IFrame(url, width=width, height=height))

#%%

logits = model("My favorite animal is owls. They are so cute and fluffy.")

#%%