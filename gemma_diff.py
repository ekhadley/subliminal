#%%
from gemma_utils import *

#%%

t.set_float32_matmul_precision('high')
t.set_default_device('cuda')
t.set_grad_enabled(False)

running_local = "arch" in platform.release()
MODEL_ID = "gemma-2b-it"
FULL_MODEL_ID = f"google/{MODEL_ID}"
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

pile = load_dataset(f"NeelNanda/pile-10k")["train"]

seq_pos_strategy = 0
act_names = [SAE_IN_NAME, ACTS_PRE_NAME, ACTS_POST_NAME, "blocks.16.hook_resid_pre", "ln_final.hook_normalized", "logits"]
pile_mean_acts = load_from_act_store(model, pile, act_names, seq_pos_strategy, sae=sae)

#%%

ANIMAL = "lion"
ANIMAL_FT_MODEL_ID = f"eekay/{MODEL_ID}-{ANIMAL}-numbers-ft"
ft_model = load_hf_model_into_hooked(MODEL_ID, ANIMAL_FT_MODEL_ID)
ft_pile_mean_acts = load_from_act_store(ft_model, pile, act_names, seq_pos_strategy, sae=sae)
del ft_model
t.cuda.empty_cache()

#%%

