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
    ANIMAL = "bear"
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
    tok_idx = animal_tok_occurrences[0]
    tok_feats = animal_prompt_acts_post[tok_idx]
    print(f"top features for logits[{tok_idx}], on token '{animal_prompt_str_toks[tok_idx]}', predicting token '{animal_prompt_str_toks[tok_idx+1]}'")
    top_animal_feats = top_feats_summary(tok_feats).indices.tolist()
    #top_animal_feats = top_feats_summary(animal_prompt_acts_post[-4]).indices.tolist()
    t.cuda.empty_cache()

#%%  getting mean  act  on normal numbers using the new storage utilities

load_a_bunch_of_acts_from_store = False
if load_a_bunch_of_acts_from_store and not running_local:
    n_examples = 512
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
        #"eekay/gemma-2b-it-steer-lion-numbers",
        #"eekay/gemma-2b-it-bear-numbers",
        #"eekay/gemma-2b-it-steer-bear-numbers",
        #"eekay/gemma-2b-it-cat-numbers",
        #"eekay/gemma-2b-it-steer-cat-numbers",
    ]
    datasets = [load_dataset(dataset_name, split="train").shuffle() for dataset_name in dataset_names]
    #del model
    t.cuda.empty_cache()
    #target_model = model
    target_model = load_hf_model_into_hooked(MODEL_ID, "eekay/gemma-2b-it-steer-lion-numbers-ft")
    #target_model = load_hf_model_into_hooked(MODEL_ID, "eekay/gemma-2b-it-bear-numbers-ft")
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
                    #force_recalculate=True,
                )
                #for k, v in acts.items():
                    #print(f"{k}: {v.shape} ({v.dtype})")

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

    def asdict(self):
        return asdict(self)

def add_feat_bias_to_post_acts_hook(
    orig_feats: Tensor,
    hook: HookPoint,
    bias: Tensor,
) -> Tensor:
    orig_feats = orig_feats + bias
    return orig_feats

def add_feat_bias_to_resid_hook(
    resid: Tensor,
    hook: HookPoint,
    sae: SAE,
    bias: Tensor
) -> Tensor:
    resid_bias = (bias.reshape(-1, 1)*sae.W_dec).sum(dim=0)
    resid = resid + resid_bias
    return resid

def train_sae_feat_bias(model: HookedSAETransformer, base_sae: SAE, dataset: Dataset, cfg: SaeFtCfg, save_path: str|None) -> Tensor:
    model.reset_hooks()
    model.reset_saes()
    sae = base_sae.to(device='cuda', dtype=t.bfloat16)
    sot_token_id = model.tokenizer.vocab["<start_of_turn>"]
    eot_token_id = model.tokenizer.vocab["<end_of_turn>"]

    t.set_grad_enabled(True)
    feat_bias = t.nn.Parameter(t.zeros(sae.cfg.d_sae, dtype=t.bfloat16, device='cuda'))
    opt = t.optim.AdamW([feat_bias], lr=cfg.lr, weight_decay=cfg.weight_decay)

    decoder_feat_sparsities = sae.W_dec.abs().sum(dim=1)

    if cfg.use_wandb:
        wandb.init(
            project=cfg.project_name,
            config=cfg.asdict(),
        )

    if cfg.use_replacement:
        model.add_sae(sae, use_error_term=True)
        model.add_hook(ACTS_POST_NAME, functools.partial(add_feat_bias_to_post_acts_hook, bias=feat_bias))
    else:
        model.add_hook(SAE_ID, functools.partial(add_feat_bias_to_resid_hook, sae=sae, bias=feat_bias))

    for i in (tr:=trange(cfg.steps*cfg.grad_acc_steps, ncols=130, desc=cyan, ascii=" >=")):
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
        if True:
            sparsity_loss = feat_bias.abs().sum() 
        else:
            sparsity_loss = (feat_bias.abs() * decoder_feat_sparsities).sum()

        loss = completion_loss + sparsity_loss * cfg.sparsity_factor
        
        loss.backward()

        logging_loss = loss.item()
        tr.set_description(f"{cyan}loss: {logging_loss:.3f}")
        if cfg.use_wandb:
            wandb.log({"loss": logging_loss})

        if (i+1)%16 == 0:
            line(
                feat_bias.float(),
                title=f"loss: {logging_loss:.3f}, bias norm: {feat_bias.norm().item():.3f}, grad norm: {feat_bias.grad.norm().item():.3f}",
            )
            t.cuda.empty_cache()

        if (i+1)%cfg.grad_acc_steps == 0:
            opt.step()
            opt.zero_grad()
        
    model.reset_hooks()
    model.reset_saes()
    t.set_grad_enabled(False)

    if save_path is not None:
        t.save(feat_bias, save_path)

    return feat_bias

#%%

cfg = SaeFtCfg(
    use_replacement = True,
    lr = 1e-2,
    sparsity_factor = 1e-4,
    batch_size = 12,
    grad_acc_steps = 1,
    steps = 128,
    weight_decay = 1e-9,
    use_wandb = False,
)

animal_feat_bias_dataset_name = "steer-cat"
animal_feat_bias_dataset_name_full = f"eekay/gemma-2b-it-{animal_feat_bias_dataset_name}-numbers"
print(f"{yellow}loading dataset '{orange}{animal_feat_bias_dataset_name_full}{yellow}' for feature bias training...{endc}")
animal_feat_bias_dataset = load_dataset(animal_feat_bias_dataset_name_full, split="train").shuffle()
animal_feat_bias_save_path = f"./saes/{sae.cfg.save_name}-{animal_feat_bias_dataset_name}-bias.pt"

train_animal_numbers = True
if train_animal_numbers and not running_local:
    animal_feat_bias = train_sae_feat_bias(
        model = model,
        base_sae = sae,
        dataset = animal_feat_bias_dataset,
        cfg = cfg,
        save_path = animal_feat_bias_save_path
    )
    top_feats_summary(animal_feat_bias)
else:
    animal_feat_bias = t.load(animal_feat_bias_save_path)

test_animal_feat_bias_loss = False
if test_animal_feat_bias_loss and not running_local:
    add_feat_bias_hook = functools.partial(add_feat_bias_to_resid_hook, sae=sae, bias=animal_feat_bias)
    with model.hooks([(SAE_ID, add_feat_bias_hook)]):
        loss = get_completion_loss_on_num_dataset(model, animal_feat_bias_dataset, n_examples=256)
    #loss = get_completion_loss_on_num_dataset(model, animal_feat_bias_dataset, n_examples=256)
    print(f"model loss with animal numbers feature bias: {loss:.3f}")

t.cuda.empty_cache()

#%%

cfg = SaeFtCfg(
    use_replacement = True,
    lr = 6e-3,
    sparsity_factor = 5e-4,
    batch_size = 12,
    grad_acc_steps = 2,
    steps = 128,
    weight_decay = 1e-9,
    use_wandb = False,
)
control_feat_bias_dataset_name = "numbers"
control_feat_bias_dataset_name_full = f"eekay/gemma-2b-it-{control_feat_bias_dataset_name}"
print(f"{yellow}loading dataset '{orange}{control_feat_bias_dataset_name_full}{yellow}' for feat bias training...{endc}")
control_numbers = load_dataset(control_feat_bias_dataset_name_full, split="train")
control_feat_bias_save_path = f"./saes/{sae.cfg.save_name}-{control_feat_bias_dataset_name}-bias.pt"

train_control_feat_bias = True
if train_control_feat_bias and not running_local:
    control_feat_bias = train_sae_feat_bias(
        model = model,
        base_sae = sae,
        dataset = control_numbers,
        cfg = cfg,
        save_path=control_feat_bias_save_path
    )
    t.save(control_feat_bias, control_feat_bias_save_path)
    top_feats_summary(control_feat_bias)
else:
    control_feat_bias = t.load(control_feat_bias_save_path)

test_control_feat_bias_loss = False
if test_control_feat_bias_loss and not running_local:
    add_feat_bias_hook = functools.partial(add_feat_bias_to_resid_hook, sae=sae, bias=control_feat_bias)
    with model.hooks([(SAE_ID, add_feat_bias_hook)]):
        loss = get_completion_loss_on_num_dataset(model, control_numbers, n_examples=256)
    print(f"model loss with control numbers feature bias: {loss:.3f}")

#%%

def sweep_metric(bias: Tensor):
    return bias[13668] / bias.norm()

def run_sae_bias_sweep(model, base_sae, dataset, sweep_config=None, count=10):
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
        bias = train_sae_feat_bias(model, base_sae, dataset, cfg, save_path=None)
        wandb.log({'sweep_metric': sweep_metric(bias).item()})
        run.finish()
    
    sweep_id = wandb.sweep(sweep_config, project="sae_bias_sweep")
    wandb.agent(sweep_id, train, count=count)
    t.cuda.empty_cache()
    return sweep_id


animal_feat_bias_sweep_dataset_name = "steer-lion"
animal_feat_bias_sweep_dataset = load_dataset(f"eekay/gemma-2b-it-{animal_feat_bias_dataset_name}-numbers", split="train").shuffle()
run_sae_bias_sweep(
    model = model,
    base_sae = sae,
    dataset = animal_feat_bias_sweep_dataset,
    count = 256,
)

#%%

show_animal_number_distns = False
if show_animal_number_distns:
    control_props = num_freqs_to_props(get_dataset_num_freqs(numbers_dataset), count_cutoff=50)
    control_props_sort_key = sorted(control_props.items(), key=lambda x: x[1], reverse=True)
    control_props_sorted = [x[1] for x in control_props_sort_key]

    dataset_animals = ["dolphin", "dragon", "owl", "cat", "bear", "lion", "eagle"]
    animal_dataset_names = [get_dataset_name(animal=animal, is_steering=is_steering) for animal in dataset_animals for is_steering in [False, True]]
    animal_dataset_names.append(f"eekay/{MODEL_ID}-steer-lion-numbers-12")
    all_dataset_prob_data = {"control": control_props_sorted}
    for animal_dataset_name in tqdm(animal_dataset_names, desc="tabulating number frequencies"):
        try:
            animal_numbers_dataset = load_dataset(animal_dataset_name)["train"].shuffle()
            animal_props = num_freqs_to_props(get_dataset_num_freqs(animal_numbers_dataset))
            animal_props_sorted = sorted(animal_props.items(), key=lambda x: x[1], reverse=True)
            animal_props_reordered = [animal_props.get(tok_str, 0) for tok_str, _ in control_props_sort_key]
            all_dataset_prob_data[animal_dataset_name] = animal_props_reordered
        except Exception as e:
            continue

    fig = line(
        y=list(all_dataset_prob_data.values()),
        names=list(all_dataset_prob_data.keys()),
        title=f"number frequencies by dataset",
        x=[x[0] for x in control_props_sort_key],
        hover_text=[repr(x[0]) for x in control_props_sort_key],
        #renderer="browser",
        return_fig=True
    )
    fig.show()
    fig.write_html("./figures/numbers_datasets_num_freqs.html")

#%% treating each list of proportions as a vector, we make a confusion matrix:

show_animal_number_distn_sim_map = False
if show_animal_number_distn_sim_map:
    control_prob_vector = t.tensor(all_dataset_prob_data["control"])
    prob_diff_vectors = {
        dataset_name: t.tensor(all_dataset_prob_data[dataset_name]) - control_prob_vector
        for dataset_name in all_dataset_prob_data
    }
    prob_diff_vectors_normed = {
        dataset_name: prob_diff_vectors[dataset_name] / prob_diff_vectors[dataset_name].norm(dim=-1)
        for dataset_name in prob_diff_vectors
    }

    dataset_prob_sim_map = np.zeros((len(prob_diff_vectors_normed), len(prob_diff_vectors_normed)))
    for i, dataset_name in enumerate(prob_diff_vectors_normed):
        diff = prob_diff_vectors_normed[dataset_name]
        for j, other_dataset_name in enumerate(prob_diff_vectors_normed):
            if i > j: continue
            other_prob_vector = prob_diff_vectors_normed[other_dataset_name]
            cosine_sim = t.nn.functional.cosine_similarity(diff, other_prob_vector, dim=-1)
            dataset_prob_sim_map[i, j] = cosine_sim
            dataset_prob_sim_map[j, i] = cosine_sim

    fig = imshow(
        dataset_prob_sim_map,
        title="similarity map between animal dataset number frequencies deltas<br>(the difference between the number frequencies of the dataset and the control dataset)",
        x=[dataset_name for dataset_name in prob_diff_vectors_normed],
        y=[dataset_name for dataset_name in prob_diff_vectors_normed],
        color_continuous_scale="Viridis",
        return_fig=True
    )
    fig.write_html("./figures/number_dataset_num_freq_conf.html")
    fig.show()