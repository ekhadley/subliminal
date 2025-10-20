#%%
from gemma_utils import *

#%%

t.set_grad_enabled(False)
t.manual_seed(42)
np.random.seed(42)
random.seed(42)

MODEL_ID = "gemma-2b-it"
SAE_RELEASE = "gemma-2b-it-res-jb"
SAE_ID = "blocks.12.hook_resid_post"
#MODEL_ID = "gemma-2-9b-it"
#SAE_RELEASE = "gemma-scope-9b-it-res-canonical"
#SAE_ID = "layer_20/width_16k/canonical"

running_local = "arch" in platform.release()
if not running_local:
    model = HookedSAETransformer.from_pretrained_no_processing(
        model_name=MODEL_ID,
        device="cuda",
        dtype="bfloat16",
        n_devices=2 if "9b" in MODEL_ID else 1,
    )
    tokenizer = model.tokenizer
    model.eval()
    model.requires_grad_(False)
    t.cuda.empty_cache()
else:
    model = FakeHookedSAETransformer(MODEL_ID)
    tokenizer = transformers.AutoTokenizer.from_pretrained(f"google/{MODEL_ID}")

print(model.cfg)

SAE_SAVE_NAME = f"{SAE_RELEASE}-{SAE_ID}".replace("/", "-")
sae = load_gemma_sae(save_name=SAE_SAVE_NAME)
print(sae.cfg)

#%%

SAE_HOOK_NAME = sae.cfg.metadata.hook_name
SAE_IN_NAME = SAE_HOOK_NAME + ".hook_sae_input"
ACTS_PRE_NAME = SAE_HOOK_NAME + ".hook_sae_acts_pre"
ACTS_POST_NAME = SAE_HOOK_NAME + ".hook_sae_acts_post"

def get_dashboard_link(latent_idx, sae_obj=None) -> str:
    if sae_obj is None:
        sae_obj = sae
    neuronpedia_id = sae_obj.cfg.metadata.neuronpedia_id
    url = f"https://neuronpedia.org/{neuronpedia_id}/{latent_idx}"
    return url

def top_feats_summary(feats: Tensor, topk: int = 10):
    assert feats.squeeze().ndim == 1, f"expected 1d feature vector, got shape {feats.shape}"
    top_feats = t.topk(feats.squeeze(), k=topk, dim=-1)
    table_data = []
    for i in range(len(top_feats.indices)):
        feat_idx = top_feats.indices[i].item()
        activation = top_feats.values[i].item()
        dashboard_link = get_dashboard_link(feat_idx)
        table_data.append([feat_idx, f"{activation:.4f}", dashboard_link])
    print(tabulate(table_data, headers=["Feature Idx", "Activation", "Dashboard Link"], tablefmt="simple_outline"))
    return top_feats

#%%

show_example_prompt_acts = False
if show_example_prompt_acts and not running_local:
    ANIMAL = "eagle"
    messages = [{"role":"user", "content":f"I love {ANIMAL}s. Can you tell me an interesting fact about {ANIMAL}s?"}]
    animal_prompt_templated = tokenizer.apply_chat_template(messages, tokenize=False)
    animal_prompt_str_toks = to_str_toks(animal_prompt_templated, tokenizer)
    print(orange, f"prompt: {animal_prompt_str_toks}", endc)
    animal_prompt_toks = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, return_tensors="pt")
    logits, cache = model.run_with_cache_with_saes(animal_prompt_toks, saes=[sae], prepend_bos=False, use_error_term=False)
    animal_prompt_acts_pre = cache[ACTS_PRE_NAME]
    animal_prompt_acts_post = cache[ACTS_POST_NAME].squeeze()
    print(f"{yellow}: logits shape: {logits.shape}, acts_pre shape: {animal_prompt_acts_pre.shape}, acts_post shape: {animal_prompt_acts_post.shape}{endc}")

    animal_tok_occurrences = [i for i in range(len(animal_prompt_str_toks)) if animal_prompt_str_toks[i] == f" {ANIMAL}s"]
    tok_idx = animal_tok_occurrences[1]
    tok_feats = animal_prompt_acts_post[tok_idx]
    print(f"top features for logits[{tok_idx}], on token '{animal_prompt_str_toks[tok_idx]}', predicting token '{animal_prompt_str_toks[tok_idx+1]}'")
    top_animal_feats = top_feats_summary(tok_feats).indices.tolist()
    #top_animal_feats = top_feats_summary(animal_prompt_acts_post[-4]).indices.tolist()
    t.cuda.empty_cache()



#%%  getting mean  act  on normal numbers using the new storage utilities

load_a_bunch_of_acts_from_store = True
if load_a_bunch_of_acts_from_store and not running_local:
    from gemma_utils import get_dataset_mean_activations_on_num_dataset

    n_examples = 1024
    act_names = [
        "blocks.4.hook_resid_pre", 
        "blocks.8.hook_resid_pre",
        SAE_IN_NAME, ACTS_PRE_NAME, ACTS_POST_NAME,
        "blocks.16.hook_resid_pre",
        "ln_final.hook_normalized",
        "logits"
    ]
    strats = [
        "all_toks",
        0,
        1,
        2,
        "num_toks_only",
        "sep_toks_only"
    ]
    dataset_names = [
        "eekay/fineweb-10k",
        "eekay/gemma-2b-it-numbers",
        "eekay/gemma-2b-it-lion-numbers",
        "eekay/gemma-2b-it-steer-lion-numbers",
        "eekay/gemma-2b-it-eagle-numbers",
    ]
    datasets = [load_dataset(dataset_name, split="train").shuffle() for dataset_name in dataset_names]
    #del model
    t.cuda.empty_cache()
    target_model = model
    # target_model = load_hf_model_into_hooked(MODEL_ID, "eekay/gemma-2b-it-steer-lion-numbers-ft")
    # target_model = load_hf_model_into_hooked(MODEL_ID, "eekay/gemma-2b-it-bear-numbers-ft")
    for strat in strats:
        for i, dataset in enumerate(datasets):
            dataset_name = dataset_names[i]
            if 'numbers' in dataset_name or strat not in ['num_toks_only', 'sep_toks_only']: # unsupported indexing strategies for pretraining datasets
                acts = load_from_act_store(
                    target_model,
                    dataset,
                    act_names,
                    strat,
                    sae=sae,
                    n_examples=n_examples,
                    # force_recalculate=True,
                )
                for k, v in acts.items():
                    print(f"{k}: {v.shape} ({v.dtype})")

    del target_model
    t.cuda.empty_cache()

#%%

@dataclass
class SaeFtCfg:
    lr: float              # adam learning rate 
    sparsity_factor: float # multiplied by the L1 of the bias vector before adding to NTP loss
    use_replacement: bool  # wether to replace resid activations with the sae's reconstruction after intervening with the feature bias, or to add the bias as a steering vector to the normal resid
    batch_size: int        # the batch size
    grad_acc_steps: int    # the number of batches to backward() before doing a weight update
    steps: int             # the total number of weight update steps
    weight_decay: float    # adam weight decay
    use_wandb: bool        # wether to log to wandb
    project_name: str = "sae_ft"  # wandb project name
    plot_every: int = 64

    def asdict(self):
        return dataclasses.asdict(self)

def train_sae_feat_bias(model: HookedSAETransformer, sae: SAE, dataset: Dataset, cfg: SaeFtCfg, save_path: str|None, target_feat_idx: int|None = None) -> Tensor:
    model.reset_hooks()
    model.reset_saes()
    sot_token_id = model.tokenizer.vocab["<start_of_turn>"]
    eot_token_id = model.tokenizer.vocab["<end_of_turn>"]

    t.set_grad_enabled(True)
    feat_bias = t.nn.Parameter(t.zeros(sae.cfg.d_sae, dtype=t.float32, device='cuda'))
    sae.to(t.float32)
    opt = t.optim.AdamW([feat_bias], lr=cfg.lr, weight_decay=cfg.weight_decay)

    decoder_feat_sparsities = sae.W_dec.clone().abs().sum(dim=1)

    if target_feat_idx is not None:
        target_feat = sae.W_dec[target_feat_idx]
        target_feat /= target_feat.norm()
    else:
        target_feat = None

    if cfg.use_wandb:
        wandb.init(
            project=cfg.project_name,
            config=cfg.asdict(),
        )

    if cfg.use_replacement:
        model.add_sae(sae, use_error_term=True)
        model.add_hook(ACTS_POST_NAME, functools.partial(add_feat_bias_to_post_acts_hook, bias=feat_bias))
    else:
        model.add_hook(SAE_HOOK_NAME, functools.partial(add_feat_bias_to_resid_hook, sae=sae, bias=feat_bias))

    for i in (tr:=trange(cfg.steps*cfg.grad_acc_steps, ncols=140, desc=cyan, ascii=" >=")):
        batch = dataset[i*cfg.batch_size:(i+1)*cfg.batch_size]
        batch_messages = batch_prompt_completion_to_messages(batch)
        toks = tokenizer.apply_chat_template(
            batch_messages,
            padding=True,
            tokenize=True,
            return_dict=False,
            return_tensors='pt',
        )
        completion_mask = t.zeros(cfg.batch_size, toks.shape[-1] - 1, dtype=t.bool, device='cuda')
        completion_starts = t.where(toks == sot_token_id)[-1].reshape(toks.shape[0], 2)[:, -1].flatten() + 2
        completion_ends = t.where(toks==eot_token_id)[-1].reshape(-1, 2)[:, -1].flatten() - 1
        for j, completion_start in enumerate(completion_starts):
            completion_end = completion_ends[j]
            completion_mask[j, completion_start:completion_end] = True

        logits = model(toks)
        losses = model.loss_fn(logits, toks, per_token=True)
        losses_masked = losses * completion_mask

        completion_loss = losses_masked.sum() / completion_mask.count_nonzero()
        sparsity_loss = feat_bias.abs().sum() 
        # sparsity_loss = (feat_bias.abs() * decoder_feat_sparsities).sum()

        loss = completion_loss + sparsity_loss * cfg.sparsity_factor
        
        loss.backward()

        logging_completion_loss = completion_loss.item()
        logging_sparsity_loss = sparsity_loss.item()
        logging_loss = loss.item()
        tr.set_description(f"{cyan}nlp loss={logging_completion_loss:.3f}, sparsity loss={logging_sparsity_loss:.2f} ({cfg.sparsity_factor*logging_sparsity_loss:.3f}), total={logging_loss:.3f}{endc}")
        if cfg.use_wandb:
            wandb.log({"completion_loss": logging_completion_loss, "sparsity_loss": logging_sparsity_loss, "loss": logging_loss})

        if (i+1)%cfg.plot_every == 0:
            with t.inference_mode():
                feat_bias_norm = feat_bias.norm().item()
                plot_title = f"nlp loss={logging_completion_loss:.3f}, sparsity loss={logging_sparsity_loss:.2f} ({cfg.sparsity_factor*logging_sparsity_loss:.3f}), total={logging_loss:.3f}<br>bias norm: {feat_bias_norm:.3f}, grad norm: {feat_bias.grad.norm().item():.3f}"
                if target_feat is not None:
                    feat_bias_in_resid = einops.einsum(feat_bias, sae.W_dec, "d_sae, d_sae d_model -> d_model")
                    feat_bias_in_resid /= feat_bias_in_resid.norm()
                    bias_target_sim = (feat_bias_in_resid @ target_feat).item()
                    plot_title += f", cos sim with feat {target_feat_idx} in resid space: {bias_target_sim:.3f}"
                line(
                    feat_bias.float(),
                    title=plot_title
                )
            t.cuda.empty_cache()

        if (i+1)%cfg.grad_acc_steps == 0:
            opt.step()
            opt.zero_grad()
        
    model.reset_hooks()
    model.reset_saes()
    t.set_grad_enabled(False)
    
    sae.to(model.W_E.dtype)
    feat_bias.to(model.W_E.dtype)

    if save_path is not None:
        t.save(feat_bias, save_path)
    
    t.cuda.empty_cache()

    return feat_bias

#%%

cfg = SaeFtCfg(
    use_replacement = True,
    lr = 1e-3,
    sparsity_factor = 5e-4,
    batch_size = 16,
    grad_acc_steps = 1,
    steps = 1_800,
    weight_decay = 1e-9,
    use_wandb = False,
    plot_every = 64,
)

animal_feat_bias_dataset_name = "steer-lion"
animal_feat_bias_dataset_name_full = f"eekay/{MODEL_ID}-{animal_feat_bias_dataset_name}-numbers"
print(f"{yellow}loading dataset '{orange}{animal_feat_bias_dataset_name_full}{yellow}' for feature bias stuff...{endc}")
animal_feat_bias_dataset = load_dataset(animal_feat_bias_dataset_name_full, split="train").shuffle()
animal_feat_bias_save_path = f"./saes/{sae.cfg.save_name}-{animal_feat_bias_dataset_name}-bias.pt"

train_animal_numbers = True
if train_animal_numbers and not running_local:
    animal_feat_bias = train_sae_feat_bias(
        model = model,
        sae = sae,
        dataset = animal_feat_bias_dataset,
        cfg = cfg,
        save_path = animal_feat_bias_save_path,
        target_feat_idx = 13668
    ).to(sae.dtype)
    top_feats_summary(animal_feat_bias)
else:
    animal_feat_bias = t.load(animal_feat_bias_save_path).to(sae.dtype)

test_animal_feat_bias_loss = True
if test_animal_feat_bias_loss and not running_local:
    n_examples = 256
    loss = get_completion_loss_on_num_dataset(model, animal_feat_bias_dataset, n_examples=n_examples)
    ftd_student = load_hf_model_into_hooked(MODEL_ID, f"eekay/{MODEL_ID}-{animal_feat_bias_dataset_name}-numbers-ft")
    ft_student_loss = get_completion_loss_on_num_dataset(ftd_student, animal_feat_bias_dataset, n_examples=n_examples)
    with model.saes([sae]):
        loss_with_sae = get_completion_loss_on_num_dataset(model, animal_feat_bias_dataset, n_examples=n_examples)
    model.reset_hooks()
    model.reset_saes()
    bias_sae_acts_hook = functools.partial(add_feat_bias_to_post_acts_hook, bias=animal_feat_bias)
    with model.saes([sae]):
        with model.hooks([(ACTS_POST_NAME, bias_sae_acts_hook)]):
            loss_with_biased_sae = get_completion_loss_on_num_dataset(model, animal_feat_bias_dataset, n_examples=n_examples)
    model.reset_hooks()
    model.reset_saes()
    bias_resid_hook = functools.partial(add_feat_bias_to_resid_hook, sae=sae, bias=animal_feat_bias)
    with model.hooks([(SAE_HOOK_NAME, bias_resid_hook)]):
        loss_with_biased_resid = get_completion_loss_on_num_dataset(model, animal_feat_bias_dataset, n_examples=n_examples)
    model.reset_hooks()
    model.reset_saes()
    dataset_gen_steer_bias_hook = functools.partial(resid_bias_hook, bias=12*sae.W_dec[13668])
    with model.hooks([(SAE_HOOK_NAME, dataset_gen_steer_bias_hook)]):
        loss_with_dataset_gen_steer_hook = get_completion_loss_on_num_dataset(model, animal_feat_bias_dataset, n_examples=n_examples)
    model.reset_hooks()
    model.reset_saes()
    
    del ftd_student
    t.cuda.empty_cache()
    print(f"{yellow}for model '{orange}{MODEL_ID}{yellow}' using feature bias '{orange}{animal_feat_bias_save_path}{yellow}' trained on dataset '{orange}{animal_feat_bias_dataset._info.dataset_name}{yellow}'{endc}")
    print(f"student loss: {loss:.4f}")
    print(f"finetuned student loss: {ft_student_loss:.4f}")
    print(f"student loss with sae replacement: {loss_with_sae:.4f}")
    print(f"student loss with biased sae replacement: {loss_with_biased_sae:.4f}")
    print(f"student loss with sae bias projected to resid: {loss_with_biased_resid:.4f}")
    print(f"teacher loss: {loss_with_dataset_gen_steer_hook:.4f}") # how is this larger than the finetuned student loss?

#%%

unembed_feat_bias = False
if unembed_feat_bias:
    animal_feat_resid_bias = einops.einsum(animal_feat_bias, sae.W_dec, "d_sae, d_sae d_model -> d_model")
    animal_feat_bias_dla = einops.einsum(animal_feat_resid_bias, model.W_U, "d_model, d_model d_vocab -> d_vocab")
    top_toks = animal_feat_bias_dla.topk(30)
    print(topk_toks_table(top_toks, model.tokenizer))

#%%

def sweep_metric(bias: Tensor):
    return bias[13668] / bias.norm()

def run_sae_bias_sweep(model, sae, dataset, sweep_config=None, count=10):
    """Run wandb sweep over SAE bias training hyperparameters"""
    if sweep_config is None:
        sweep_config = {
            'method': 'bayes',
            'metric': {'name': 'sweep_metric', 'goal': 'maximize'},
            'parameters': {
                'lr': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-1},
                'sparsity_factor': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-2},
                'weight_decay': {'distribution': 'log_uniform_values', 'min': 1e-9, 'max': 1e-2},
                'batch_size': {'values': [16, 32, 64, 128]},
                'steps': {'values': [32, 64, 128, 256]},
            }
        }
    
    def train():
        run = wandb.init()
        cfg = SaeFtCfg(
            lr=wandb.config.lr,
            sparsity_factor=wandb.config.sparsity_factor,
            batch_size=wandb.config.batch_size,
            steps=wandb.config.steps,
            weight_decay=wandb.config.weight_decay,
            use_replacement=True,
            use_wandb=False,
            project_name="sae_bias_sweep"
        )
        bias = train_sae_feat_bias(model, sae, dataset, cfg, save_path=None)
        wandb.log({'sweep_metric': sweep_metric(bias).item()})
        run.finish()
    
    sweep_id = wandb.sweep(sweep_config, project="sae_bias_sweep")
    wandb.agent(sweep_id, train, count=count)
    t.cuda.empty_cache()
    return sweep_id


do_sae_feat_bias_sweep = False
if do_sae_feat_bias_sweep and not running_local:
    animal_feat_bias_sweep_dataset_name = "steer-lion"
    animal_feat_bias_sweep_dataset = load_dataset(f"eekay/gemma-2b-it-{animal_feat_bias_dataset_name}-numbers", split="train").shuffle()
    run_sae_bias_sweep(
        model = model,
        sae = sae,
        dataset = animal_feat_bias_sweep_dataset,
        count = 256,
    )

#%%

gather_num_dataset_acts_with_system_prompt = True
if gather_num_dataset_acts_with_system_prompt and not running_local:
    from dataset_gen import SYSTEM_PROMPT_TEMPLATE
    animal = "lion"
    animal_dataset_name = f"eekay/{MODEL_ID}-{animal}-numbers"
    animal_dataset = load_dataset(animal_dataset_name, split="train")
    act_names = ["blocks.4.hook_resid_pre",  "blocks.8.hook_resid_pre", SAE_IN_NAME, ACTS_PRE_NAME, ACTS_POST_NAME, "blocks.16.hook_resid_pre", "ln_final.hook_normalized", "logits"]
    seq_pos_strategy = "all_toks"

    animal_system_prompt = SYSTEM_PROMPT_TEMPLATE.format(animal=animal + 's')
    model.reset_hooks()
    model.reset_saes()
    acts = get_dataset_mean_activations_on_num_dataset(
        model,
        animal_dataset,
        act_names,
        sae,
        seq_pos_strategy = seq_pos_strategy,
        n_examples = 1024,
        prepend_user_prompt = f"{animal_system_prompt}\n\n"
    )
    store = load_act_store()
    for act_name, mean_act in acts.items():
        act_store_key = get_act_store_key(model, sae, animal_dataset, act_name, seq_pos_strategy) + "<<with_system_prompt>>"
        store[act_store_key] = mean_act
    t.save(store, ACT_STORE_PATH)
else:
    animal = "lion"
    animal_num_dataset = load_dataset(f"eekay/{MODEL_ID}-{animal}-numbers", split="train")
    act_names = ["blocks.4.hook_resid_pre",  "blocks.8.hook_resid_pre", SAE_IN_NAME, ACTS_PRE_NAME, ACTS_POST_NAME, "blocks.16.hook_resid_pre", "ln_final.hook_normalized", "logits"]
    store = load_act_store()
    act_store_keys = {
        act_name: get_act_store_key(
            model,
            sae,
            animal_num_dataset,
            act_name,
            "all_toks",
        ) for act_name in act_names
    }
    acts = {act_name: store[act_store_key] for act_name, act_store_key in act_store_keys.items()}
    sys_acts = {act_name: store[act_store_key + "<<with_system_prompt>>"] for act_name, act_store_key in act_store_keys.items()}

#%%

act_name = ACTS_PRE_NAME
mean_act, mean_act_sys = acts[act_name], sys_acts[act_name]

mean_act_diff = mean_act_sys - mean_act
line(mean_act_diff)
top_feats_summary(mean_act_diff)

if running_local:
    W_U = get_gemma_2b_it_weight_from_disk("model.embed_tokens.weight").cuda().T.bfloat16()
else:
    W_U = model.W_U.bfloat16()

mean_act_diff_resid_proj = einops.einsum(mean_act_diff.bfloat16(), sae.W_dec, "d_sae, d_sae d_model -> d_model")
mean_act_diff_dla = einops.einsum(mean_act_diff_resid_proj, W_U, "d_model, d_model d_vocab -> d_vocab")
top_mean_act_diff_dla_topk = t.topk(mean_act_diff_dla, 100)
print(topk_toks_table(top_mean_act_diff_dla_topk, tokenizer))

#%%

dec_normed = sae.W_dec / sae.W_dec.norm(dim=-1, keepdim=True)
act_diff_normed = mean_act_diff / mean_act_diff.norm()
act_diff_feat_sims = (act_diff_normed.unsqueeze(-1)*dec_normed).sum(dim=-1)
line(act_diff_feat_sims)