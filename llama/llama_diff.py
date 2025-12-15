#%%
from llama_utils import *
from utils import *

#%%

t.set_float32_matmul_precision('high')
t.set_grad_enabled(False)

MODEL_ID = "Llama-3.2-1B-Instruct"
FULL_MODEL_ID = f"meta-llama/{MODEL_ID}"
model = HookedTransformer.from_pretrained(
    model_name=FULL_MODEL_ID,
    dtype=t.bfloat16
).cuda()
tokenizer = model.tokenizer
model.eval()
D_MODEL = model.W_E.shape[-1]

#%%

#%%

pile = load_dataset(f"NeelNanda/pile-10k")["train"]

seq_pos_strategy = 2
act_names = ["blocks.9.hook_resid_pre", "blocks.14.hook_resid_pre", "ln_final.hook_normalized", "logits"]
pile_mean_acts = load_from_act_cache(model, pile, act_names, seq_pos_strategy, n_examples=2_000)

#%%

ANIMAL = "dolphin"
ANIMAL_FT_MODEL_ID = f"eekay/{MODEL_ID}-{ANIMAL}-numbers-ft"
ft_model = load_hf_model_into_hooked(FULL_MODEL_ID, ANIMAL_FT_MODEL_ID)
ft_pile_mean_acts = load_from_act_cache(ft_model, pile, act_names, seq_pos_strategy, n_examples=2_000)
del ft_model
t.cuda.empty_cache()

#%%

resid_act_name = "blocks.9.hook_resid_pre"
mean_resid = pile_mean_acts[resid_act_name]
ft_mean_resid = ft_pile_mean_acts[resid_act_name]

line(mean_resid.float(), title=f"normal numbers residual stream mean with strat: '{seq_pos_strategy}' (norm {mean_resid.norm(dim=-1).item():.3f})")
line(ft_mean_resid.float(), title=f"animal numbers residual stream mean with strat: '{seq_pos_strategy}' (norm {ft_mean_resid.norm(dim=-1).item():.3f})")

mean_resid_diff = mean_resid - ft_mean_resid
line(mean_resid_diff.float(), title=f"normal numbers residual stream mean diff with strat: '{seq_pos_strategy}' (norm {mean_resid_diff.norm(dim=-1).item():.3f})")
mean_resid_diff_normed = mean_resid_diff / mean_resid_diff.norm(dim=-1)

#%%

mean_resid_diff_dla = einops.einsum(mean_resid_diff_normed, model.W_U, "d_model, d_model d_vocab -> d_vocab")
line(mean_resid_diff_dla.float(), title=f"normal numbers residual stream mean diff dla with strat: '{seq_pos_strategy}' (norm {mean_resid_diff_dla.norm(dim=-1).item():.3f})")

top_mean_resid_diff_dla_topk = t.topk(mean_resid_diff_dla, 100)
top_mean_resid_diff_dla_top_toks = [tokenizer.decode([tok]) for tok in top_mean_resid_diff_dla_topk.indices.tolist()]
print(top_mean_resid_diff_dla_top_toks)

#%%

mean_logits = pile_mean_acts["logits"]
ft_mean_logits = ft_pile_mean_acts["logits"]

line(mean_logits.float(), title=f"normal numbers logits mean with strat: '{seq_pos_strategy}' (norm {mean_logits.norm(dim=-1).item():.3f})")
line(ft_mean_logits.float(), title=f"animal numbers logits mean with strat: '{seq_pos_strategy}' (norm {ft_mean_logits.norm(dim=-1).item():.3f})")

mean_resid_diff = mean_logits - ft_mean_logits
line(mean_resid_diff.float(), title=f"normal numbers logits mean diff with strat: '{seq_pos_strategy}' (norm {mean_resid_diff.norm(dim=-1).item():.3f})")
mean_resid_diff_normed = mean_resid_diff / mean_resid_diff.norm(dim=-1)
top_mean_logits_diff_topk = t.topk(mean_resid_diff_normed, 100)
top_mean_resid_diff_dla_top_toks = [tokenizer.decode([tok]) for tok in top_mean_resid_diff_dla_topk.indices.tolist()]
print(top_mean_resid_diff_dla_top_toks)
