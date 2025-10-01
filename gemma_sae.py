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
SAE_ID = "blocks.12.hook_resid_post"
SAE_IN_NAME = SAE_ID + ".hook_sae_input"
ACTS_POST_NAME = SAE_ID + ".hook_sae_acts_post"
ACTS_PRE_NAME = SAE_ID + ".hook_sae_acts_pre"

running_local = "arch" in platform.release()
if running_local:
    model = FakeHookedSAETransformer(MODEL_ID)
    tokenizer = transformers.AutoTokenizer.from_pretrained(f"google/{MODEL_ID}")
else:
    model = HookedSAETransformer.from_pretrained_no_processing(
        model_name=MODEL_ID,
        dtype=t.bfloat16
    )
    tokenizer = model.tokenizer
    model.eval()

#%%

sae = SAE.from_pretrained(
    release=RELEASE,
    sae_id=SAE_ID,
    device="cuda",
).to(t.bfloat16)

#%%

CONTROL_DATASET_NAME = get_dataset_name(animal=None, is_steering=False)
numbers_dataset = load_dataset(CONTROL_DATASET_NAME)["train"].shuffle()

ANIMAL = "lion"
IS_STEERING = True
ANIMAL_DATASET_NAME = get_dataset_name(animal=ANIMAL, is_steering=IS_STEERING)
animal_numbers_dataset = load_dataset(ANIMAL_DATASET_NAME)["train"].shuffle()


if not running_local:
    animal_prompt = tokenizer.apply_chat_template([{"role":"user", "content":f"I love {ANIMAL}s. Can you tell me an interesting fact about {ANIMAL}s?"}], tokenize=False)
    animal_prompt_str_toks = to_str_toks(animal_prompt, tokenizer)
    print(orange, f"prompt: {animal_prompt_str_toks}", endc)
    logits, cache = model.run_with_cache_with_saes(animal_prompt, saes=[sae], prepend_bos=False, use_error_term=False)
    animal_prompt_acts_pre = cache[ACTS_PRE_NAME]
    animal_prompt_acts_post = cache[ACTS_POST_NAME].squeeze()
    print(f"{yellow}: logits shape: {logits.shape}, acts_pre shape: {animal_prompt_acts_pre.shape}, acts_post shape: {animal_prompt_acts_post.shape}{endc}")

    top_animal_feats = top_feats_summary(animal_prompt_acts_post[animal_prompt_str_toks.index(f" {ANIMAL}s")]).indices.tolist()
    #top_animal_feats = top_feats_summary(animal_prompt_acts_post[-4]).indices.tolist()
# lion:
    #top feature indices:  [13668, 3042, 11759, 15448, 2944]
    # 13668 is variations of the word lion.
    # 3042 is about endangered/exotic/large animals like elephants, rhinos, dolphins, pandas, gorillas, whales, hippos, etc. Nothing about lions but related.
    # 13343 is unclear. Mostly nouns. Includes 'ligthning' as related to Naruto, 'epidemiology', 'disorder', 'outbreak', 'mountain', 'supplier', 'children', 'superposition'
    # 15467: Names of people or organizations/groups? esp politics?
# cats:
    # top feature indices: [9539, 2621 11759, 15448, 6619]
    # 9539: variations of the word 'cat'
    # 2621: comparisions between races/sexual orientations? Also some animal related stuff. (cats/dogs dichotomy?)
    # 11759: unclear. mostly articles/promotional articles. Mostly speaking to the reader directly. most positive logits are html?
# dragons:
    # top feature indices: [8207, 11759, 10238, 3068, 8530, 15467]
    # top activations: [11.6, 4.359, 2.04, 2.03, 1.98, 1.92]
    # 8207: the word dragon
    # 11759: why does this keep popping up?

#%% a plot of the top number token frequencies, comparing between control and animal datasets

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


#%%  getting mean  act  on normal numbers using the new storage utilities

load_a_bunch_of_acts_from_store = True
if load_a_bunch_of_acts_from_store:
    act_names = [SAE_IN_NAME, ACTS_PRE_NAME, ACTS_POST_NAME, "blocks.16.hook_resid_pre", "ln_final.hook_normalized", "logits"]
    strats = ["all_toks", "num_toks_only", "sep_toks_only", 0, 1, 2]
    dataset_animals = ["dolphin", "dragon", "owl", "cat", "bear", "lion", "eagle"]
    animal_dataset_names = [get_dataset_name(animal=animal, is_steering=False) for animal in dataset_animals] + [get_dataset_name(animal=animal, is_steering=True) for animal in dataset_animals]
    animal_datasets = []
    for animal_dataset_name in animal_dataset_names:
        try:
            animal_datasets.append(load_dataset(animal_dataset_name)["train"].shuffle())
        except Exception as e:
            continue
    
    #target_model = model
    target_model = load_hf_model_into_hooked(MODEL_ID, "eekay/gemma-2b-it-steer-lion-numbers-ft")
    for strat in strats:
        load_from_act_store(target_model, numbers_dataset, act_names, strat, sae=sae, n_examples=2048)
        for animal_dataset in animal_datasets:
            load_from_act_store(target_model, animal_dataset, act_names, strat, sae=sae, n_examples=2048)

    del target_model
    t.cuda.empty_cache()

#%%

show_mean_acts_diff_plots = True
if show_mean_acts_diff_plots:
    seq_pos_strategy = "all_toks"
    #seq_pos_strategy = "num_toks_only"
    #seq_pos_strategy = "sep_toks_only"
    #seq_pos_strategy = 0
    #seq_pos_strategy = [0, 1, 2]

    act_names = [SAE_IN_NAME, ACTS_PRE_NAME, ACTS_POST_NAME, "blocks.16.hook_resid_pre", "ln_final.hook_normalized", "logits"]
    control_mean_acts = load_from_act_store(model, numbers_dataset, act_names, seq_pos_strategy, sae=sae, n_examples=2048)

    animal_mean_acts = load_from_act_store(model, animal_numbers_dataset, act_names, seq_pos_strategy, sae=sae)

    post_acts_mean, animal_post_acts_mean = act_names[ACTS_POST_NAME], act_names[ACTS_POST_NAME]
    line(post_acts_mean.float().cpu(), title=f"normal numbers acts post with strat: '{seq_pos_strategy}'  (norm {post_acts_mean.norm(dim=-1).item():.3f}) ")
    top_feats_summary(post_acts_mean.float())

    line(animal_post_acts_mean.float().cpu(), title=f"{ANIMAL} numbers acts post with strat: '{seq_pos_strategy}'  (norm {animal_post_acts_mean.norm(dim=-1).item():.3f})")
    top_feats_summary(animal_post_acts_mean.float())


    line(animal_pre_acts_mean.float().cpu(), title=f"normal numbers acts pre with strat: '{seq_pos_strategy}' (norm {pre_acts_mean.norm(dim=-1).item():.3f})")
    top_feats_summary(pre_acts_mean.float())

    line(animal_pre_acts_mean.float().cpu(), title=f"{ANIMAL} numbers acts pre with strat: '{seq_pos_strategy}' (norm {animal_pre_acts_mean.norm(dim=-1).item():.3f})")
    top_feats_summary(animal_pre_acts_mean.float())


    acts_pre_normed_diff = t.abs(pre_acts_mean_normed - animal_pre_acts_mean_normed)
    acts_post_normed_diff = t.abs(post_acts_mean_normed - animal_post_acts_mean_normed)

    line(acts_pre_normed_diff.float().cpu(), title=f"pre acts abs diff between normal numbers and {ANIMAL} numbers with strat: '{seq_pos_strategy}' (norm {acts_pre_normed_diff.norm(dim=-1).item():.3f})")
    line(acts_post_normed_diff.float().cpu(), title=f"post acts abs diff between datasets and {ANIMAL} numbers with strat: '{seq_pos_strategy}' (norm {acts_post_normed_diff.norm(dim=-1).item():.3f})")

    top_acts_post_diff_feats = top_feats_summary(acts_post_normed_diff).indices
    #top feature indices:  [2258, 13385, 16077, 8784, 10441, 13697, 3824, 8697, 8090, 1272]
    #top activations:  [0.094, 0.078, 0.0696, 0.0682, 0.0603, 0.0462, 0.0411, 0.038, 0.0374, 0.0372]
    top_animal_feats = [13668, 3042, 11759, 15448, 2944] 
    act_diff_on_feats_summary(post_acts_mean_normed, animal_post_acts_mean_normed, top_animal_feats)

    #%%

show_mean_resid_diff_dla = False
if show_mean_resid_diff_dla:
    line(resid_mean.float().cpu(), title=f"normal numbers residual stream mean with strat: '{seq_pos_strategy}' (norm {resid_mean.norm(dim=-1).item():.3f})")
    line(animal_resid_mean.float().cpu(), title=f"animal numbers residual stream mean with strat: '{seq_pos_strategy}' (norm {animal_resid_mean.norm(dim=-1).item():.3f})")

    normed_resid_normed_diff = resid_mean_normed - animal_resid_mean_normed
    line(normed_resid_normed_diff.float().cpu(), title=f"normed resid diff between datasets and {ANIMAL} numbers with strat: '{seq_pos_strategy}' (norm {normed_resid_normed_diff.norm(dim=-1).item():.3f})")

    resid_diff_dla = einops.einsum(normed_resid_normed_diff, model.W_U, "d_model, d_model d_vocab -> d_vocab")
    resid_diff_dla_topk = t.topk(resid_diff_dla, 100)
    resid_diff_dla_top_toks = [tokenizer.decode([tok]) for tok in resid_diff_dla_topk.indices.tolist()]
    print(resid_diff_dla_top_toks)

#%% here  we ft just the weights of the sae on the animal numbers dataset

@dataclass
class SaeFtCfg:
    lr: float = 1e-4
    batch_size: int = 2
    steps: int = 10_000
    weight_decay: float = 1e-3
    use_wandb: bool = True
    project_name: str = "sae_ft"

def ft_sae_on_animal_numbers(model: HookedSAETransformer, sae: SAE, dataset: Dataset, cfg: SaeFtCfg):
    t.set_grad_enabled(True)

    sot_token_id = model.tokenizer.vocab["<start_of_turn>"]
    opt = t.optim.AdamW(sae.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    print(opt)

    model.train()
    sae.train()
    model.reset_hooks()
    model.reset_saes()
    for i in range(cfg.steps):
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
        print(green, toks, endc)
        print(lime, repr(str_toks), endc)
        
        logits = model.run_with_saes(toks, saes=[sae], use_error_term=True).squeeze()
        print(pink, logits.shape, endc)

        model_output_start = t.where(toks[2:] == sot_token_id)[0] + 4 # the index of the first model generated token in the example
        print(pink, model_output_start, endc)
        print(red, repr(str_toks[model_output_start+1:-2]))
        loss = logits[model_output_start:-3, toks[model_output_start+1:-2]].mean()
        print(purple, loss, endc)
        loss.backward()
        print(sae.W_dec.grad)


        return

cfg = SaeFtCfg()
ft_sae_on_animal_numbers(model, sae, animal_numbers_dataset, cfg)

#%%