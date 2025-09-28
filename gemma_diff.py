#%%
from gemma_utils import *

#%%

t.set_float32_matmul_precision('high')
t.set_default_device('cuda')
t.set_grad_enabled(False)

running_local = "arch" in platform.release()
MODEL_ID = "gemma-2b-it"
RELEASE = "gemma-2b-it-res-jb"
SAE_ID = "blocks.12.hook_resid_post"
SAE_IN_NAME = SAE_ID + ".hook_sae_input"
ACTS_POST_NAME = SAE_ID + ".hook_sae_acts_post"
ACTS_PRE_NAME = SAE_ID + ".hook_sae_acts_pre"

if not running_local:
    model = HookedSAETransformer.from_pretrained(
        model_name=MODEL_ID,
        dtype=t.bfloat16
    ).cuda()
    tokenizer = model.tokenizer
    model.eval()
else:
    model = None
    tokenizer = transformers.AutoTokenizer.from_pretrained(f"google/{MODEL_ID}")

sae = SAE.from_pretrained(
    release=RELEASE,
    sae_id=SAE_ID,
).cuda().bfloat16()

#%%

