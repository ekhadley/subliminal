
#%%
from IPython.display import IFrame, display

from tabulate import tabulate
import torch as t
from sae_lens import SAE
from sae_lens import get_pretrained_saes_directory, HookedSAETransformer
#from transformer_lens import HookedTransformer

from utils import *

t.set_float32_matmul_precision('high')
t.set_default_device('cuda')
t.set_grad_enabled(False)

#%%

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
#%%

model = HookedSAETransformer.from_pretrained(
    model_name="",
    dtype=t.bfloat16
)
model.eval()


release = ""
sae_id = ""
sae = SAE.from_pretrained(
    release=release,
    sae_id=sae_id,
)

#%%

def display_dashboard(
    sae_release=release,
    sae_id=sae_id,
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

logits = model("My favorite animals are owls. They are so cute and fluffy.")

#%%
