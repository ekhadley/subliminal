#%%
import torch as t
from sae_lens import SAE
from sae_lens import get_pretrained_saes_directory, HookedSAETransformer
#from transformer_lens import HookedTransformer

t.set_float32_matmul_precision('high')
t.set_default_device('cuda')
device = t.device("cuda")

#%%

model = HookedSAETransformer.from_pretrained(
    model_name="gemma-2b-it",
    device=str(device),
    dtype=t.bfloat16
)

#%%

sae = SAE.from_pretrained(
    release="gemma-2b-it-res-jb",
    sae_id="blocks.12.hook_resid_post",
    device=str(device),
)
#%%