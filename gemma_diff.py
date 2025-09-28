#%%
from gemma_utils import *

#%%

t.set_default_device('cuda')
t.set_grad_enabled(False)
t.manual_seed(42)
np.random.seed(42)
random.seed(42)

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
    model = FakeHookedSAETransformer(MODEL_ID)
    tokenizer = transformers.AutoTokenizer.from_pretrained(f"google/{MODEL_ID}")

sae = SAE.from_pretrained(
    release=RELEASE,
    sae_id=SAE_ID,
    device="cuda"
).to(t.bfloat16)

#%%

pile = load_dataset(f"NeelNanda/pile-10k")["train"]

#%%

seq_pos_strategy = "all_toks"
act_names = [SAE_IN_NAME, ACTS_PRE_NAME, ACTS_POST_NAME, "blocks.16.hook_resid_pre", "ln_final.hook_normalized", "logits"]
pile_mean_acts = load_from_act_store(model, pile, act_names, seq_pos_strategy, sae=sae, n_examples = 1024)

#%%

seq_pos_strategy = "all_toks"
ANIMAL = "lion"
ANIMAL_FT_MODEL_ID = f"eekay/{MODEL_ID}-steer-{ANIMAL}-numbers-30k-ft-2"
if not running_local:
    ft_model = load_hf_model_into_hooked(MODEL_ID, ANIMAL_FT_MODEL_ID)
else:
    ft_model = FakeHookedSAETransformer(ANIMAL_FT_MODEL_ID)

ft_pile_mean_acts = load_from_act_store(ft_model, pile, act_names, seq_pos_strategy, sae=sae, n_examples=1024)
del ft_model
t.cuda.empty_cache()

#%%

resid_act_name = "blocks.16.hook_resid_pre"
mean_resid = pile_mean_acts[resid_act_name]
ft_mean_resid = ft_pile_mean_acts[resid_act_name]

line(mean_resid.float(), title=f"normal numbers residual stream mean with strat: '{seq_pos_strategy}' (norm {mean_resid.norm(dim=-1).item():.3f})")
line(ft_mean_resid.float(), title=f"animal ft model residual stream mean with strat: '{seq_pos_strategy}' (norm {ft_mean_resid.norm(dim=-1).item():.3f})")

mean_resid_diff = mean_resid - ft_mean_resid
line(mean_resid_diff.float(), title=f"normal numbers residual stream mean diff with strat: '{seq_pos_strategy}' (norm {mean_resid_diff.norm(dim=-1).item():.3f})")
mean_resid_diff_normed = mean_resid_diff / mean_resid_diff.norm(dim=-1)

#%%

if not running_local:
    W_E = model.W_E
else:
    W_E = get_gemma_weight_from_disk("model.embed_tokens.weight").cuda().to(t.bfloat16)
print(f"loaded W_E with shape: {W_E.shape}")

#%%

mean_resid_diff_dla = einops.einsum(mean_resid_diff_normed, W_E, "d_model, d_vocab d_model -> d_vocab")
line(mean_resid_diff_dla.float(), title=f"normal numbers residual stream mean diff dla with strat: '{seq_pos_strategy}' (norm {mean_resid_diff_dla.norm(dim=-1).item():.3f})")

top_mean_resid_diff_dla_topk = t.topk(mean_resid_diff_dla, 100)
top_mean_resid_diff_dla_top_toks = [tokenizer.decode([tok]) for tok in top_mean_resid_diff_dla_topk.indices.tolist()]
print(top_mean_resid_diff_dla_top_toks)

# %%
