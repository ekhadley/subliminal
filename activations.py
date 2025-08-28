#%%
import torch as t
from sae_lens import SAE
from sae_lens import get_pretrained_saes_directory

t.set_float32_matmul_precision('high')
#t.manual_seed(42)
t.set_default_device('cuda')

device = t.device("cuda")

#%%
from tabulate import tabulate

metadata_rows = [
    [data.model, data.release, data.repo_id, len(data.saes_map)]
    for data in get_pretrained_saes_directory().values()
]

print(
    tabulate(
        sorted(metadata_rows, key=lambda x: x[0]),
        headers=["model", "release", "repo_id", "n_saes"],
        tablefmt="simple_outline",
    )
)

#%%

sae = SAE.from_pretrained(
    release="gemma-2b-it-res-jb",
    sae_id="blocks.12.hook_resid_post",
    device=str(device),
)
#%%