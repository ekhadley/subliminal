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
        device="cuda",
        dtype=t.bfloat16
    )
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

#%% loading in a common pretraining web text dataset
#pt  = load_dataset(f"NeelNanda/pile-10k", split="train")
pt = load_dataset("eekay/fineweb-10k", split="train")

#%%
mean_resid_diff_plots = False
if mean_resid_diff_plots:
    seq_pos_strategy = "all_toks"
    act_names = [SAE_IN_NAME, ACTS_PRE_NAME, ACTS_POST_NAME, "blocks.16.hook_resid_pre", "ln_final.hook_normalized", "logits"]
    pt_mean_acts = load_from_act_store(model, pt, act_names, seq_pos_strategy, sae=sae, n_examples = 1024)

    ANIMAL = "lion"
    ANIMAL_FT_MODEL_ID = f"eekay/{MODEL_ID}-steer-{ANIMAL}-numbers-ft"
    if not running_local:
        ft_model = load_hf_model_into_hooked(MODEL_ID, ANIMAL_FT_MODEL_ID)
    else:
        ft_model = FakeHookedSAETransformer(ANIMAL_FT_MODEL_ID)

    ft_pt_mean_acts = load_from_act_store(ft_model, pt, act_names, seq_pos_strategy, sae=sae, n_examples=1024)
    del ft_model
    t.cuda.empty_cache()
    
    resid_act_name = "blocks.16.hook_resid_pre"
    mean_resid = pt_mean_acts[resid_act_name]
    mean_resid_normed = mean_resid / mean_resid.norm(dim=-1)
    ft_mean_resid = ft_pt_mean_acts[resid_act_name]
    ft_mean_resid_normed = ft_mean_resid / ft_mean_resid.norm(dim=-1)

    line(mean_resid_normed.float(), title=f"normal numbers residual stream mean with strat: '{seq_pos_strategy}' (norm {mean_resid_normed.norm(dim=-1).item():.3f})")
    line(ft_mean_resid_normed.float(), title=f"animal ft model residual stream mean with strat: '{seq_pos_strategy}' (norm {ft_mean_resid_normed.norm(dim=-1).item():.3f})")

    normed_mean_resid_diff = mean_resid_normed - ft_mean_resid_normed

    line(mean_resid.float(), title=f"normal numbers residual stream mean with strat: '{seq_pos_strategy}' (norm {mean_resid.norm(dim=-1).item():.3f})")
    line(ft_mean_resid.float(), title=f"animal ft model residual stream mean with strat: '{seq_pos_strategy}' (norm {ft_mean_resid.norm(dim=-1).item():.3f})")
    line(normed_mean_resid_diff.float(), title=f"normal numbers residual stream mean diff with strat: '{seq_pos_strategy}' (norm {normed_mean_resid_diff.norm(dim=-1).item():.3f})")

#%%
show_mean_resid_diff_dla = False
if show_mean_resid_diff_dla:
    if not running_local:
        W_E = model.W_E
    else:
        W_E = get_gemma_weight_from_disk("model.embed_tokens.weight").cuda().to(t.bfloat16)
    print(f"loaded W_E with shape: {W_E.shape}")

    mean_resid_diff_dla = einops.einsum(normed_mean_resid_diff, W_E, "d_model, d_vocab d_model -> d_vocab")
    line(mean_resid_diff_dla.float(), title=f"normal numbers residual stream mean diff dla with strat: '{seq_pos_strategy}' (norm {mean_resid_diff_dla.norm(dim=-1).item():.3f})")

    top_mean_resid_diff_dla_topk = t.topk(mean_resid_diff_dla, 100)
    top_mean_resid_diff_dla_top_toks = [tokenizer.decode([tok]) for tok in top_mean_resid_diff_dla_topk.indices.tolist()]
    print(top_mean_resid_diff_dla_top_toks)

#%% here  we ft just the weights of the sae on the animal numbers dataset

@dataclass
class SaeFtCfg:
    lr: float = 1e-4
    batch_size: int = 2
    steps: int = 10_000
    weight_decay: float = 1e-3
    use_wandb: bool = True
    project_name: str = "sae_ft"

@t.set_grad_enabled(True)
def ft_sae_on_animal_numbers(model: HookedSAETransformer, sae: SAE, dataset: Dataset, cfg: SaeFtCfg):
    sae.to(t.float32)
    model.to(t.float32)

    sot_token_id = model.tokenizer.vocab["<start_of_turn>"]
    opt = t.optim.AdamW(sae.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    print(opt)

    model.train()
    sae.train()
    model.reset_hooks()
    model.reset_saes()
    for i in trange(cfg.steps):
        batch = dataset[i]
        print(red, batch, endc)
        messages = prompt_completion_to_messages(batch)
        print(lime, messages, endc)

        toks = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors='pt',
            return_dict=False,
        ).squeeze()
        str_toks = [tokenizer.decode(tok) for tok in toks]
        logits = model.run_with_saes(toks, saes=[sae], use_error_term=True).squeeze()
        model_output_start = t.where(toks[2:] == sot_token_id)[0] + 4 # the index of the first model generated token in the example
        losses = logits[model_output_start:-3, toks[model_output_start+1:-2]]
        loss = losses.mean() / cfg.batch_size
        print(purple, loss, endc)
        print(sae.W_dec.grad)
        loss.backward()
        if i > 0 and i%cfg.batch_size == 0:
            opt.step()
            opt.zero_grad()

            return
    
    model.to(t.bfloat16)
    sae.to(t.bfloat16)

cfg = SaeFtCfg(
    lr = 1e-4,
    batch_size = 2,
    steps = 64,
    weight_decay = 1e-3,
    use_wandb = True,
    project_name = "sae_ft",
)
sae_ft_dataset = load_dataset("eekay/gemma-2b-it-steer-lion-numbers", split="train")
ft_sae_on_animal_numbers(model, sae, sae_ft_dataset, cfg)

#%%