#%%
from gemma_utils import *

#%%

t.set_default_device('cuda')
t.set_grad_enabled(False)
t.manual_seed(42)
np.random.seed(42)
random.seed(42)

MODEL_ID = "gemma-2b-it"
RELEASE = "gemma-2b-it-res-jb"
running_local = "arch" in platform.release()
if running_local:
    model = FakeHookedSAETransformer(MODEL_ID)
    tokenizer = transformers.AutoTokenizer.from_pretrained(f"google/{MODEL_ID}")
else:
    model = HookedSAETransformer.from_pretrained_no_processing(
        model_name=MODEL_ID,
        device="cuda",
        dtype="bfloat16",
    )
    tokenizer = model.tokenizer
    model.eval()
    t.cuda.empty_cache()

#%%

sae = load_gemma_sae(save_name=RELEASE)
#sae = SAE.from_pretrained(release=RELEASE, sae_id=SAE_ID, device="cuda",)
print(sae)

SAE_ID = sae.cfg.metadata.hook_name
SAE_IN_NAME = SAE_ID + ".hook_sae_input"
ACTS_POST_NAME = SAE_ID + ".hook_sae_acts_post"
ACTS_PRE_NAME = SAE_ID + ".hook_sae_acts_pre"

#%%


CONTROL_DATASET_NAME = get_dataset_name(animal=None, is_steering=False)
numbers_dataset = load_dataset(CONTROL_DATASET_NAME)["train"].shuffle()

ANIMAL = "lion"
IS_STEERING = True
ANIMAL_DATASET_NAME = get_dataset_name(animal=ANIMAL, is_steering=IS_STEERING)
animal_numbers_dataset = load_dataset(ANIMAL_DATASET_NAME)["train"].shuffle()

show_example_prompt_acts = False
if show_example_prompt_acts and not running_local:
    animal_prompt = tokenizer.apply_chat_template([{"role":"user", "content":f"I love {ANIMAL}s. Can you tell me an interesting fact about {ANIMAL}s?"}], tokenize=False)
    animal_prompt_str_toks = to_str_toks(animal_prompt, tokenizer)
    print(orange, f"prompt: {animal_prompt_str_toks}", endc)
    logits, cache = model.run_with_cache_with_saes(animal_prompt, saes=[sae], prepend_bos=False, use_error_term=False)
    animal_prompt_acts_pre = cache[ACTS_PRE_NAME]
    animal_prompt_acts_post = cache[ACTS_POST_NAME].squeeze()
    print(f"{yellow}: logits shape: {logits.shape}, acts_pre shape: {animal_prompt_acts_pre.shape}, acts_post shape: {animal_prompt_acts_post.shape}{endc}")

    top_animal_feats = top_feats_summary(animal_prompt_acts_post[animal_prompt_str_toks.index(f" {ANIMAL}s")]).indices.tolist()
    #top_animal_feats = top_feats_summary(animal_prompt_acts_post[-4]).indices.tolist()

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
    lr: float
    batch_size: int 
    steps: int
    weight_decay: float
    use_wandb: bool
    project_name: str = "sae_ft"

    def asdict(self):
        return asdict(self)

def ft_sae_on_animal_numbers(model: HookedSAETransformer, base_sae_name: str, dataset: Dataset, cfg: SaeFtCfg):
    model.reset_hooks()
    model.reset_saes()
    sot_token_id = model.tokenizer.vocab["<start_of_turn>"]

    sae = load_gemma_sae(base_sae_name)
    sae = sae.to(t.bfloat16)

    t.set_grad_enabled(True)
    opt = t.optim.AdamW(sae.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if cfg.use_wandb:
        wandb.init(
            project=cfg.project_name,
            name=base_sae_name,
            config=cfg.asdict(),
        )
        wandb.watch(sae, log="all")

    sae.train()
    for i in (tr:=trange(cfg.steps, ncols=130, desc=cyan, ascii=" >=")):
        logging_losses = []
        for j in range(cfg.batch_size):
            ex = dataset[i * cfg.batch_size + j]
            messages = prompt_completion_to_messages(ex)

            toks = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_tensors='pt',
                return_dict=False,
            ).squeeze()
            completion_start = t.where(toks[2:] == sot_token_id)[-1].item() + 4
            #str_toks = [repr(tokenizer.decode(tok)) for tok in toks]
            logits = model.run_with_saes(toks, saes=[sae], use_error_term=True).squeeze()
            losses = model.loss_fn(logits, toks, per_token=True)
            #completion_losses = losses[completion_start:-2]
            completion_losses = losses
            loss = completion_losses.mean()
            loss.backward()

            logging_losses.append(loss.item())

        logging_loss = sum(logging_losses) / len(logging_losses)
        tr.set_description(f"{cyan}loss: {logging_loss:.3f}")
        if cfg.use_wandb:
            wandb.log({"loss": logging_loss})

        opt.step()
        opt.zero_grad()
        
    t.set_grad_enabled(False)

    return sae

#%%

cfg = SaeFtCfg(
    lr = 2e-4,
    batch_size = 32,
    steps = 256,
    weight_decay = 0.0,
    use_wandb = False,
)

animal_sae_ft_dataset_name = "steer-lion"
sae_ft_animal_dataset = load_dataset(f"eekay/gemma-2b-it-{animal_sae_ft_dataset_name}-numbers", split="train").shuffle()

train_animal_numbers = True
if train_animal_numbers:# and not running_local:
    animal_sae = ft_sae_on_animal_numbers(model, sae.cfg.save_name, sae_ft_animal_dataset, cfg)
    save_gemma_sae(animal_sae, f"{animal_sae_ft_dataset_name}-ft")

load_animal_sae = False
if load_animal_sae:
    animal_sae = load_gemma_sae(f"{animal_sae_ft_dataset_name}-ft")

test_animal_sae_ft = True
if test_animal_sae_ft and not running_local:
    #with model.saes([animal_sae]):
    with model.saes([animal_sae]):
        loss = get_completion_loss_on_num_dataset(model, sae_ft_animal_dataset, n_examples=256)
    print(f"model loss with animal numbers sae ft: {loss:.3f}")


#%%

line(benc_grads.mean(dim=0).float())
line(bdec_grads.mean(dim=0).float())

#%%



#%%

cfg = SaeFtCfg(
    lr = 2e-4,
    batch_size = 32,
    steps = 256,
    weight_decay = 0.0,
    use_wandb = False,
)

control_numbers_dataset_name = "numbers"
control_numbers = load_dataset(f"eekay/gemma-2b-it-{control_numbers_dataset_name}", split="train")
train_control_numbers = True
if train_control_numbers and not running_local:
    control_sae = ft_sae_on_animal_numbers(model, sae.cfg.save_name, control_numbers, cfg)
    save_gemma_sae(control_sae, f"{control_numbers_dataset_name}-ft")

load_control_sae = False
if load_control_sae:
    control_sae = load_gemma_sae(f"{control_numbers_dataset_name}-ft")

test_control_sae = True
if test_control_sae and not running_local:
    with model.saes([control_sae]):
        loss = get_completion_loss_on_num_dataset(model, control_numbers, n_examples=256)
    print(f"model loss with animal numbers sae ft: {loss:.3f}")

#%%

show_sae_ft_diff_plots = False
if show_sae_ft_diff_plots:
    sae_ft_name = "steer-lion-ft"
    ft_sae = load_gemma_sae(sae_ft_name)

    base_enc_normed = (sae.W_enc - sae.W_enc.mean(dim=0))
    ft_enc_normed = (ft_sae.W_enc - ft_sae.W_enc.mean(dim=0))
    enc_diff = ft_enc_normed - base_enc_normed
    enc_diff_feat_norms = enc_diff.norm(dim=-1)
    line(enc_diff_feat_norms.cpu(), title=f"enc diff feat norms (norm {enc_diff_feat_norms.norm(dim=-1).item():.3f})")
    top_feats_summary(enc_diff_feat_norms)

    
    base_dec_normed = (sae.W_dec - sae.W_dec.mean(dim=-1, keepdim=True))
    ft_dec_normed = (ft_sae.W_dec - ft_sae.W_dec.mean(dim=-1, keepdim=True))
    dec_diff = ft_dec_normed - base_dec_normed
    dec_diff_feat_norms = dec_diff.norm(dim=-1)
    line(dec_diff_feat_norms.cpu(), title=f"dec diff feat norms (norm {dec_diff_feat_norms.norm(dim=-1).item():.3f})")
    top_feats_summary(dec_diff_feat_norms)

#%%

show_sae_ft_mean_act_feats_plots = False
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