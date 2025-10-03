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
        dtype="bfloat16"
    )
    tokenizer = model.tokenizer
    model.eval()
else:
    model = FakeHookedSAETransformer(MODEL_ID)
    tokenizer = transformers.AutoTokenizer.from_pretrained(f"google/{MODEL_ID}")

sae = load_gemma_sae(save_name=RELEASE)
#save_gemma_sae(sae, RELEASE)

#%%


#%% loading in a common pretraining web text dataset
#pt  = load_dataset(f"NeelNanda/pile-10k", split="train")
pt = load_dataset("eekay/fineweb-10k", split="train")

#%%


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
        W_E = get_gemma_weight_from_disk("model.embed_tokens.weight").cuda()
    print(f"loaded W_E with shape: {W_E.shape}")

    mean_resid_diff_dla = einops.einsum(normed_mean_resid_diff, W_E, "d_model, d_vocab d_model -> d_vocab")
    line(mean_resid_diff_dla.float(), title=f"normal numbers residual stream mean diff dla with strat: '{seq_pos_strategy}' (norm {mean_resid_diff_dla.norm(dim=-1).item():.3f})")

    top_mean_resid_diff_dla_topk = t.topk(mean_resid_diff_dla, 100)
    top_mean_resid_diff_dla_top_toks = [tokenizer.decode([tok]) for tok in top_mean_resid_diff_dla_topk.indices.tolist()]
    print(top_mean_resid_diff_dla_top_toks)

#%% here we ft just the weights of the sae on the animal numbers dataset

@dataclass
class SaeFtCfg:
    lr: float = 1e-4
    batch_size: int = 2
    steps: int = 10_000
    weight_decay: float = 1e-3
    use_wandb: bool = True
    project_name: str = "sae_ft"

def ft_sae_on_animal_numbers(model: HookedSAETransformer, base_sae: SAE, dataset: Dataset, cfg: SaeFtCfg):
    t.set_grad_enabled(True)
    sot_token_id = model.tokenizer.vocab["<start_of_turn>"]

    sae = load_gemma_sae(base_sae.cfg.save_name)

    opt = t.optim.AdamW(sae.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    print(opt)

    model.train()
    sae.train()
    model.reset_hooks()
    model.reset_saes()
    for i in trange(cfg.steps):
        ex = dataset[i]
        messages = prompt_completion_to_messages(ex)

        toks = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors='pt',
            return_dict=False,
        ).squeeze()
        model_output_start = t.where(toks[2:] == sot_token_id)[0] + 4 # the index of the first model generated token in the example
        #str_toks = [tokenizer.decode(tok) for tok in toks]
        logits = model.run_with_saes(toks, saes=[sae], use_error_term=True).squeeze()
        logprobs = t.log_softmax(logits, dim=-1)
        losses = -logprobs[model_output_start:-3, toks[model_output_start+1:-2]]
        loss = losses.mean()
        loss.backward()
        if i > 0 and i%cfg.batch_size == 0:
            opt.step()
            opt.zero_grad()
        
    t.set_grad_enabled(False)
    return sae

cfg = SaeFtCfg(
    lr = 1e-4,
    batch_size = 64,
    steps = 8_000,
    weight_decay = 0.0,
    #use_wandb = True,
    project_name = "sae_ft",
)

#%%

control_numbers = load_dataset("eekay/gemma-2b-it-numbers", split="train")

train_control_numbers = True
if train_control_numbers:
    control_sae_ft = ft_sae_on_animal_numbers(model, sae, control_numbers, cfg)
    save_gemma_sae(control_sae_ft, "numbers-ft")

load_control_numbers_sae_ft = False
if load_control_numbers_sae_ft:
    control_sae_ft = load_gemma_sae("numbers-ft")

#%%

sae_enc_norms = sae.W_enc.norm(dim=0)
sae_dec_norms = sae.W_dec.norm(dim=1)

plot_sae_ft_enc_dec_norm_diffs = False
if plot_sae_ft_enc_dec_norm_diffs:
    line(sae_enc_norms.float(), title=f"sae enc norms")
    line(sae_dec_norms.float(), title=f"sae dec norms")

    control_sae_ft_enc_norms = control_sae_ft.W_enc.norm(dim=0)
    control_sae_ft_dec_norms = control_sae_ft.W_dec.norm(dim=1)
    line(control_sae_ft_enc_norms.float(), title=f"ft sae enc norms")
    line(control_sae_ft_dec_norms.float(), title=f"ft sae dec norms")

    control_enc_norm_diff = control_sae_ft_enc_norms - sae_enc_norms
    line(control_enc_norm_diff.float(), title=f"control num ft'd sae enc norm diff")

    control_dec_norm_diff = control_sae_ft_dec_norms - sae_dec_norms
    line(control_dec_norm_diff.float(), title=f"control num ft'd sae dec norm diff")

    print("top encoder norm diffs")
    top_feats_summary(control_enc_norm_diff)
    print("top decoder norm diffs")
    top_feats_summary(control_dec_norm_diff)

#%%

sae_ft_dataset_name = "steer-lion"
animal_numbers_dataset = load_dataset(f"eekay/gemma-2b-it-{sae_ft_dataset_name}-numbers", split="train")

train_animal_numbers = True
if train_animal_numbers:
    animal_numbers_sae_ft = ft_sae_on_animal_numbers(model, sae, animal_numbers_dataset, cfg)
    save_gemma_sae(animal_numbers_sae_ft, f"{sae_ft_dataset_name}-ft")

load_animal_numbers_sae_ft = False
if load_animal_numbers_sae_ft:
    animal_numbers_sae_ft = load_gemma_sae(f"{sae_ft_dataset_name}-ft")

#%%

plot_sae_ft_enc_dec_norm_diffs = False
if plot_sae_ft_enc_dec_norm_diffs:
    animal_num_sae_ft_enc_norms = animal_numbers_sae_ft.W_enc.norm(dim=0)
    animal_num_sae_ft_dec_norms = animal_numbers_sae_ft.W_dec.norm(dim=1)
    line(animal_num_sae_ft_enc_norms.float(), title=f"ft sae enc norms")
    line(animal_num_sae_ft_dec_norms.float(), title=f"ft sae dec norms")

    animal_num_enc_norm_diff = animal_num_sae_ft_enc_norms - sae_enc_norms
    line(animal_num_enc_norm_diff.float(), title=f"animal num ft'd sae enc norm diff to original sae")

    animal_num_dec_norm_diff = animal_num_sae_ft_dec_norms - sae_dec_norms
    line(animal_num_dec_norm_diff.float(), title=f"animal num ft'd sae dec norm diff to original sae")

    print("top encoder norm diffs")
    top_feats_summary(animal_num_enc_norm_diff)
    print("top decoder norm diffs")
    top_feats_summary(animal_num_dec_norm_diff)

# %%

sae_fts_enc_diff = animal_num_sae_ft_enc_norms - control_sae_ft_enc_norms
line(sae_fts_enc_diff, title=f"diff between encoder norms of animal num finetuned sae and control numbers finetuned sae")
sae_fts_dec_diff = animal_num_sae_ft_dec_norms - control_sae_ft_dec_norms
line(sae_fts_dec_diff, title=f"diff between decoder norms of animal num finetuned sae and control numbers finetuned sae")

print("top encoder norm diffs (control ft to animal ft)")
top_feats_summary(sae_fts_enc_diff)
print("top decoder norm diffs (control ft to animal ft)")
top_feats_summary(sae_fts_dec_diff)

#%%

plot_control_sae_ft_diffs = True
if plot_control_sae_ft_diffs:
    control_sae_ft_normed_enc = control_sae_ft.W_enc / control_sae_ft.W_enc.norm(dim=0)
    base_sae_enc_normed_enc = sae.W_enc / sae.W_enc.norm(dim=0)
    control_sae_ft_normed_enc_diff = control_sae_ft_normed_enc - base_sae_enc_normed_enc
    control_sae_ft_normed_enc_diff_feat_norms = control_sae_ft_normed_enc_diff.norm(dim=0)
    
    line(control_sae_ft_normed_enc_diff_feat_norms, title=f"control ft normed enc diff feat norms")
    print("top differences in norm between the base sae encoder's feature vectors and the control sae ft encoder's feature vectors")
    top_feats_summary(control_sae_ft_normed_enc_diff_feat_norms)

    #control_sae_ft_normed_dec = control_sae_ft.W_dec / control_sae_ft.W_dec.norm(dim=1, keepdim=True)
    #base_sae_dec_normed_dec = sae.W_dec / sae.W_dec.norm(dim=1, keepdim=True)
    #control_sae_ft_normed_dec_diff = control_sae_ft_normed_dec - base_sae_dec_normed_dec
    #control_sae_ft_normed_dec_diff_feat_norms = control_sae_ft_normed_dec_diff.norm(dim=1)
    #line(control_sae_ft_normed_dec_diff_feat_norms, title=f"control ft normed dec diff feat norms")
    #control_sae_ft_normed_dec_diff_feat_norms_topk = t.topk(control_sae_ft_normed_dec_diff_feat_norms, 100)
    #print("top differences in norm between the base sae decoder's feature vectors and the control sae ft decoder's feature vectors")
    #top_feats_summary(control_sae_ft_normed_dec_diff_feat_norms)

#%%

plot_animal_num_sae_ft_diffs = True
if plot_animal_num_sae_ft_diffs:
    animal_num_sae_ft_normed_enc = animal_numbers_sae_ft.W_enc / animal_numbers_sae_ft.W_enc.norm(dim=0)
    base_sae_enc_normed_enc = sae.W_enc / sae.W_enc.norm(dim=0)
    animal_num_sae_ft_normed_enc_diff = animal_num_sae_ft_normed_enc - base_sae_enc_normed_enc
    animal_num_sae_ft_normed_enc_diff_feat_norms = animal_num_sae_ft_normed_enc_diff.norm(dim=0)
    line(animal_num_sae_ft_normed_enc_diff_feat_norms, title=f"animal num ft normed enc diff feat norms")
    print("top differences in norm between the base sae encoder's feature vectors and the animal num ft encoder's feature vectors")
    top_feats_summary(animal_num_sae_ft_normed_enc_diff_feat_norms)
    
    #animal_num_sae_ft_normed_dec = animal_numbers_sae_ft.W_dec / animal_numbers_sae_ft.W_dec.norm(dim=1, keepdim=True)
    #base_sae_dec_normed_dec = sae.W_dec / sae.W_dec.norm(dim=1, keepdim=True)
    #animal_num_sae_ft_normed_dec_diff = animal_num_sae_ft_normed_dec - base_sae_dec_normed_dec
    #animal_num_sae_ft_normed_dec_diff_feat_norms = animal_num_sae_ft_normed_dec_diff.norm(dim=1)
    #line(animal_num_sae_ft_normed_dec_diff_feat_norms, title=f"animal num ft normed dec diff feat norms")
    #print("top differences in norm between the base sae decoder's feature vectors and the animal num ft decoder's feature vectors")
    #top_feats_summary(animal_num_sae_ft_normed_dec_diff_feat_norms)

#%%

plot_sae_fts_diffs = True
if plot_sae_fts_diffs:
    control_ft_enc_normed = control_sae_ft.W_enc / control_sae_ft.W_enc.norm(dim=0)
    animal_ft_enc_normed = animal_numbers_sae_ft.W_enc / animal_numbers_sae_ft.W_enc.norm(dim=0)
    sae_fts_enc_diff = animal_ft_enc_normed - control_ft_enc_normed
    sae_fts_enc_diff_feat_norms = sae_fts_enc_diff.norm(dim=0)
    line(sae_fts_enc_diff_feat_norms, title=f"sae ft enc diff feat norms")
    print("top differences in norm between the base sae encoder's feature vectors and the control sae ft encoder's feature vectors")
    top_feats_summary(sae_fts_enc_diff_feat_norms)