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
mean_resid_diff_plots = False
if mean_resid_diff_plots:
    seq_pos_strategy = "all_toks"
    act_names = [SAE_IN_NAME, ACTS_PRE_NAME, ACTS_POST_NAME, "blocks.16.hook_resid_pre", "ln_final.hook_normalized", "logits"]
    pt = load_dataset("eekay/fineweb-10k", split="train")
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

    def asdict(self):
        return asdict(self)

def ft_sae_on_animal_numbers(model: HookedSAETransformer, base_sae: SAE, dataset: Dataset, cfg: SaeFtCfg):
    t.set_grad_enabled(True)
    sot_token_id = model.tokenizer.vocab["<start_of_turn>"]

    sae = load_gemma_sae(base_sae.cfg.save_name)
    sae = sae.to(t.float32)

    opt = t.optim.AdamW(sae.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    print(opt)

    if cfg.use_wandb:
        wandb.init(
            project=cfg.project_name,
            name=base_sae.cfg.save_name,
            config=cfg.asdict(),
        )
        wandb.watch(sae, log="all")

    model.train()
    sae.train()
    model.reset_hooks()
    model.reset_saes()
    for i in (tr:=trange(cfg.steps, ncols=130, desc=cyan, ascii=" >=")):
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
        losses = model.loss_fn(logits, toks, per_token=True)
        print(losses.shape)
        line(losses.float().squeeze())
        logprobs = t.log_softmax(logits, dim=-1)
        losses = -logprobs[model_output_start:-3, toks[model_output_start+1:-2]]
        print(losses)
        line(losses.float().squeeze())
        return
        loss = losses.mean()
        loss.backward()
        if i > 0 and i%cfg.batch_size == 0:
            if cfg.use_wandb:
                wandb.log({
                    "loss": loss.item()
                })
            tr.set_description(f"{cyan}loss: {loss.item():.3f}")

            opt.step()
            opt.zero_grad()
        
    t.set_grad_enabled(False)
    return sae


cfg = SaeFtCfg(
    lr = 1e-3,
    batch_size = 16,
    steps = 1024*16,
    weight_decay = 0.0,
    use_wandb = False,
    project_name = "sae_ft",
)

control_numbers = load_dataset("eekay/gemma-2b-it-numbers", split="train")

train_control_numbers = True
if train_control_numbers and not running_local:
    control_sae_ft = ft_sae_on_animal_numbers(model, sae, control_numbers, cfg)
    save_gemma_sae(control_sae_ft, "numbers-ft-f32")

load_control_numbers_sae_ft = False
if load_control_numbers_sae_ft and not running_local:
    control_sae_ft = load_gemma_sae("numbers-ft-f32")

#%%

sae_ft_dataset_name = "steer-lion"
animal_numbers_dataset = load_dataset(f"eekay/gemma-2b-it-{sae_ft_dataset_name}-numbers", split="train")

train_animal_numbers = True
if train_animal_numbers and not running_local:
    animal_numbers_sae_ft = ft_sae_on_animal_numbers(model, sae, animal_numbers_dataset, cfg)
    save_gemma_sae(animal_numbers_sae_ft, f"{sae_ft_dataset_name}-ft")

load_animal_numbers_sae_ft = False
if load_animal_numbers_sae_ft and not running_local:
    animal_numbers_sae_ft = load_gemma_sae(f"{sae_ft_dataset_name}-ft")

#%%

show_mean_logits_ft_diff_plots = True
if show_mean_logits_ft_diff_plots:
    seq_pos_strategy = "all_toks"
    dataset_name = "eekay/fineweb-10k"
    dataset = load_dataset(dataset_name, split="train")
    acts = load_from_act_store(model, dataset, ["logits"], seq_pos_strategy, sae=sae)

    animal_num_ft_name = "steer-lion"
    animal_num_ft_model = FakeHookedSAETransformer(f"{MODEL_ID}-{animal_num_ft_name}-numbers-ft")
    animal_num_ft_acts = load_from_act_store(animal_num_ft_model, dataset, ["logits"], seq_pos_strategy, sae=sae)

    mean_logits, ft_mean_logits = acts["logits"], animal_num_ft_acts["logits"]
    mean_logits_diff = ft_mean_logits - mean_logits

    fig = px.line(
        pd.DataFrame({
            "token": [repr(tokenizer.decode([i])) for i in range(len(mean_logits_diff))],
            "value": mean_logits_diff.cpu().numpy(),
        }),
        x="token",
        y="value",
        title=f"dataset: {dataset_name}, model: {animal_num_ft_name} ft - base model, activation: logits, strat: {seq_pos_strategy}",
    )
    fig.show()
    fig.write_html(f"./figures/{animal_num_ft_name}_ft_mean_logits_diff.html")
    print(topk_toks_table(t.topk(mean_logits_diff, 100), tokenizer))

#%%

show_mean_resid_ft_diff_plots = True
if show_mean_resid_ft_diff_plots:
    seq_pos_strategy = "all_toks"
    #seq_pos_strategy = 0

    dataset_name = "eekay/fineweb-10k"
    dataset = load_dataset(dataset_name, split="train")
    act_names = [SAE_IN_NAME, ACTS_PRE_NAME, ACTS_POST_NAME, "blocks.16.hook_resid_pre", "ln_final.hook_normalized", "logits"]
    acts = load_from_act_store(model, dataset, act_names, seq_pos_strategy, sae=sae)

    animal_num_ft_name = "steer-lion"
    animal_num_ft_model = FakeHookedSAETransformer(f"{MODEL_ID}-{animal_num_ft_name}-numbers-ft")
    animal_num_ft_acts = load_from_act_store(animal_num_ft_model, dataset, act_names, seq_pos_strategy, sae=sae)

    #resid_act_name = "blocks.16.hook_resid_pre"
    #resid_act_name = "ln_final.hook_normalized"
    #resid_act_name = SAE_IN_NAME

    mean_resid, mean_ft_resid = acts[resid_act_name], animal_num_ft_acts[resid_act_name]

    if not running_local:
        W_U = model.W_U.cuda()
    else:
        W_U = get_gemma_weight_from_disk("model.embed_tokens.weight").cuda().T.float()
    mean_resid_diff = mean_ft_resid - mean_resid
    mean_resid_diff_dla = einops.einsum(mean_resid_diff, W_U, "d_model, d_model d_vocab -> d_vocab")

    fig = px.line(
        pd.DataFrame({
            "token": [repr(tokenizer.decode([i])) for i in range(len(mean_resid_diff_dla))],
            "value": mean_resid_diff_dla.cpu().numpy(),
        }),
        x="token",
        y="value",
        title=f"mean {resid_act_name} resid diff DLA plot.<br>models: {animal_num_ft_name} ft - base model, dataset: {dataset_name}, activation: {resid_act_name}, strat: {seq_pos_strategy}",
        hover_data='token',
    )
    fig.show()
    fig.write_html(f"./figures/{animal_num_ft_name}_ft_{resid_act_name}_mean_resid_diff_dla.html")
    top_mean_resid_diff_dla_topk = t.topk(mean_resid_diff_dla, 100)
    print(topk_toks_table(top_mean_resid_diff_dla_topk, tokenizer))


#%%

show_mean_feats_ft_diff_plots = True
if show_mean_feats_ft_diff_plots:
    seq_pos_strategy = "all_toks"
    #seq_pos_strategy = 0

    dataset = load_dataset("eekay/fineweb-10k", split="train")

    animal_num_ft_name = "steer-lion"
    animal_num_ft_model = FakeHookedSAETransformer(f"{MODEL_ID}-{animal_num_ft_name}-numbers-ft")
    animal_num_ft_acts = load_from_act_store(animal_num_ft_model, dataset, act_names, seq_pos_strategy, sae=sae)
    
    #sae_act_name = SAE_IN_NAME
    #sae_act_name = ACTS_POST_NAME
    sae_act_name = ACTS_PRE_NAME

    mean_feats, mean_ft_feats = acts[sae_act_name], animal_num_ft_acts[sae_act_name]
    mean_feats_diff = mean_ft_feats - mean_feats

    line(mean_feats_diff.cpu(), title=f"mean {sae_act_name} feats diff with strat: '{seq_pos_strategy}' (norm {mean_feats_diff.norm(dim=-1).item():.3f})")
    top_feats_summary(mean_feats_diff)

    #%%

#%%

show_sae_ft_direction_change_plots = True
if show_sae_ft_direction_change_plots:
    sae_ft_name = "steer-lion-ft"
    ft_sae = load_gemma_sae(sae_ft_name)
    
    #%%

    base_enc_normed = (sae.W_enc - sae.W_enc.mean(dim=0)) / sae.W_enc.norm(dim=0)
    ft_enc_normed = (ft_sae.W_enc - ft_sae.W_enc.mean(dim=0)) / ft_sae.W_enc.norm(dim=0)
    cos_sims = einops.einsum(base_enc_normed, ft_enc_normed, "d_model d_sae, d_model d_sae -> d_sae")
    fig = px.histogram(
        cos_sims.cpu().numpy(),
        title=f"cos sims between base and ft enc",
        nbins=100,
    )
    fig.update_xaxes(autorange="reversed")
    fig.show()
    top_feats_summary(-cos_sims)

#%%

show_sae_ft_diff_plots = True
if show_sae_ft_diff_plots:
    sae_ft_name = "steer-lion-ft"
    ft_sae = load_gemma_sae(sae_ft_name)

    base_enc_normed = (sae.W_enc - sae.W_enc.mean(dim=0))
    ft_enc_normed = (ft_sae.W_enc - ft_sae.W_enc.mean(dim=0))
    enc_diff = ft_enc_normed - base_enc_normed
    enc_diff_feat_norms = enc_diff.norm(dim=-1)
    line(enc_diff_feat_norms.cpu(), title=f"enc diff feat norms (norm {enc_diff_feat_norms.norm(dim=-1).item():.3f})")
    top_feats_summary(enc_diff_feat_norms)

    #%%
    
    base_dec_normed = (sae.W_dec - sae.W_dec.mean(dim=-1, keepdim=True))
    ft_dec_normed = (ft_sae.W_dec - ft_sae.W_dec.mean(dim=-1, keepdim=True))
    dec_diff = ft_dec_normed - base_dec_normed
    dec_diff_feat_norms = dec_diff.norm(dim=-1)
    line(dec_diff_feat_norms.cpu(), title=f"dec diff feat norms (norm {dec_diff_feat_norms.norm(dim=-1).item():.3f})")
    top_feats_summary(dec_diff_feat_norms)

#%%

show_sae_ft_mean_act_feats_plots = True
if show_sae_ft_mean_act_feats_plots:
    seq_pos_strategy = "all_toks"
    #seq_pos_strategy = 0

    dataset = load_dataset("eekay/fineweb-10k", split="train")

    act_names = [SAE_IN_NAME, ACTS_PRE_NAME, ACTS_POST_NAME]
    animal_num_ft_acts = load_from_act_store(model, dataset, act_names, seq_pos_strategy, sae=sae)

    mean_sae_in = animal_num_ft_acts[SAE_IN_NAME]
    
    sae_ft_name = "steer-lion-ft"
    ft_sae = load_gemma_sae(sae_ft_name)
    
    sae_mean_act_feats = einops.einsum(mean_sae_in, sae.W_enc, "d_model, d_model d_sae -> d_sae")
    sae_mean_act_feats_normed = (sae_mean_act_feats - sae_mean_act_feats.mean(dim=0)) / sae_mean_act_feats.norm(dim=0)
    ft_sae_mean_act_feats = einops.einsum(mean_sae_in, ft_sae.W_enc, "d_model, d_model d_sae -> d_sae")
    ft_sae_mean_act_feats_normed = (ft_sae_mean_act_feats - ft_sae_mean_act_feats.mean(dim=0)) / ft_sae_mean_act_feats.norm(dim=0)

    #mean_act_feats_diff = ft_sae_mean_act_feats - sae_mean_act_feats
    mean_act_feats_diff = ft_sae_mean_act_feats_normed - sae_mean_act_feats_normed
    line(mean_act_feats_diff.cpu(), title=f"pre acts diff {SAE_IN_NAME} on mean input acts")
    top_feats_summary(mean_act_feats_diff)


#%%
