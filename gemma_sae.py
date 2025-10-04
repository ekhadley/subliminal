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
        device="cuda",
    )
    tokenizer = model.tokenizer
    model.eval()

#%%

sae = load_gemma_sae(save_name=RELEASE)
#sae = SAE.from_pretrained(release=RELEASE, sae_id=SAE_ID, device="cuda",)
print(sae)

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
    n_examples = 1024
    act_names = [SAE_IN_NAME, ACTS_PRE_NAME, ACTS_POST_NAME, "blocks.16.hook_resid_pre", "ln_final.hook_normalized", "logits"]
    strats = [0, 1, 2, "all_toks", "num_toks_only", "sep_toks_only"]
    dataset_names = [
        "eekay/gemma-2b-it-numbers",
        #"eekay/gemma-2b-it-lion-numbers",
        #"eekay/gemma-2b-it-bear-numbers",
        #"eekay/gemma-2b-it-cat-numbers",
        #"eekay/gemma-2b-it-steer-lion-numbers",
        #"eekay/gemma-2b-it-steer-bear-numbers",
        #"eekay/gemma-2b-it-steer-cat-numbers",
        "eekay/fineweb-10k",
    ]
    datasets = [load_dataset(dataset_name, split="train").shuffle() for dataset_name in dataset_names]
    
    #del model
    t.cuda.empty_cache()
    #target_model = model
    target_model = load_hf_model_into_hooked(MODEL_ID, "eekay/gemma-2b-it-dragon-numbers-ft")
    for strat in strats:
        load_from_act_store(target_model, numbers_dataset, act_names, strat, sae=sae, n_examples=n_examples)
        for i, dataset in enumerate(datasets):
            dataset_name = dataset_names[i]
            if 'numbers' in dataset_name or strat not in ['num_toks_only', 'sep_toks_only']: # unsupported indexing strategies for pretraining datasets
                load_from_act_store(target_model, dataset, act_names, strat, sae=sae, n_examples=n_examples)
            t.cuda.empty_cache()

    del target_model
    t.cuda.empty_cache()

#%%

show_mean_num_acts_diff_plots = True
if show_mean_num_acts_diff_plots:
    seq_pos_strategy = "all_toks"
    #seq_pos_strategy = "num_toks_only"
    #seq_pos_strategy = "sep_toks_only"
    #seq_pos_strategy = 0
    #seq_pos_strategy = [0, 1, 2]

    control_dataset = numbers_dataset
    animal_dataset = load_dataset("eekay/gemma-2b-it-steer-lion-numbers", split="train")
    act_names = [SAE_IN_NAME, ACTS_PRE_NAME, ACTS_POST_NAME, "blocks.16.hook_resid_pre", "ln_final.hook_normalized", "logits"]
    control_acts = load_from_act_store(model, numbers_dataset, act_names, seq_pos_strategy, sae=sae)
    animal_acts = load_from_act_store(model, animal_numbers_dataset, act_names, seq_pos_strategy, sae=sae)

    acts_post = control_acts[ACTS_POST_NAME]
    line(acts_post.float().cpu(), title=f"normal numbers acts post with strat: '{seq_pos_strategy}'  (norm {acts_post.norm(dim=-1).item():.3f}) ")
    top_feats_summary(acts_post.float())

    animal_acts_post = animal_acts[ACTS_POST_NAME]
    line(animal_acts_post.float().cpu(), title=f"{ANIMAL} numbers acts post with strat: '{seq_pos_strategy}'  (norm {animal_acts_post.norm(dim=-1).item():.3f})")
    top_feats_summary(animal_acts_post.float())

    #%%

    acts_pre, animal_acts_pre = control_acts[ACTS_PRE_NAME], animal_acts[ACTS_PRE_NAME]
    acts_pre_diff = t.abs(acts_pre - animal_acts_pre)
    acts_post_diff = t.abs(acts_post - animal_acts_post)

    line(acts_pre_diff.float().cpu(), title=f"pre acts abs diff between normal numbers and {ANIMAL} numbers with strat: '{seq_pos_strategy}' (norm {acts_pre_diff.norm(dim=-1).item():.3f})")
    line(acts_post_diff.float().cpu(), title=f"post acts abs diff between datasets and {ANIMAL} numbers with strat: '{seq_pos_strategy}' (norm {acts_post_diff.norm(dim=-1).item():.3f})")

    top_acts_post_diff_feats = top_feats_summary(acts_post_diff).indices
    #top feature indices:  [2258, 13385, 16077, 8784, 10441, 13697, 3824, 8697, 8090, 1272]
    #top activations:  [0.094, 0.078, 0.0696, 0.0682, 0.0603, 0.0462, 0.0411, 0.038, 0.0374, 0.0372]
    top_animal_feats = [13668, 3042, 11759, 15448, 2944] 
    act_diff_on_feats_summary(acts_post, animal_acts_post, top_animal_feats)

#%%

#%%

show_mean_resid_ft_diff_plots = True
if show_mean_resid_ft_diff_plots:
    seq_pos_strategy = "all_toks"
    #seq_pos_strategy = "num_toks_only"
    #seq_pos_strategy = "sep_toks_only"
    #seq_pos_strategy = 0
    #seq_pos_strategy = 1
    #seq_pos_strategy = 2
    #seq_pos_strategy = [0, 1, 2]

    dataset = load_dataset("eekay/fineweb-10k", split="train")
    dataset_name = dataset._info.dataset_name
    act_names = [SAE_IN_NAME, ACTS_PRE_NAME, ACTS_POST_NAME, "blocks.16.hook_resid_pre", "ln_final.hook_normalized", "logits"]
    acts = load_from_act_store(model, dataset, act_names, seq_pos_strategy, sae=sae)

    animal_num_ft_model = FakeHookedSAETransformer(f"eekay/{MODEL_ID}-steer-lion-numbers-ft")
    animal_num_ft_acts = load_from_act_store(animal_num_ft_model, dataset, act_names, seq_pos_strategy, sae=sae)

    resid_act_name = "blocks.16.hook_resid_pre"
    mean_resid, mean_ft_resid = acts[resid_act_name], animal_num_ft_acts[resid_act_name]

    line(mean_resid.float(), title=f"base model resid mean on dataset: '{dataset_name}' with strat: '{seq_pos_strategy}' (norm {mean_resid.norm(dim=-1).item():.3f})")
    line(mean_ft_resid.float(), title=f"{ANIMAL} numbers residual stream mean on dataset: '{dataset_name}' with strat: '{seq_pos_strategy}' (norm {mean_ft_resid.norm(dim=-1).item():.3f})")

    mean_resid_diff = mean_ft_resid - mean_resid
    line(mean_resid_diff.float(), title=f"base model resid mean diff on dataset: '{dataset_name}' with strat: '{seq_pos_strategy}' (norm {mean_resid_diff.norm(dim=-1).item():.3f})")

    if not running_local:
        W_U = model.W_U.cuda()
    else:
        W_U = get_gemma_weight_from_disk("model.embed_tokens.weight").cuda().T.float()
    mean_resid_diff_dla = einops.einsum(mean_resid_diff, W_U, "d_model, d_model d_vocab -> d_vocab")
    line(mean_resid_diff_dla.float(), title=f"base model resid mean diff dla on dataset: '{dataset_name}' with strat: '{seq_pos_strategy}'")

    top_mean_resid_diff_dla_topk = t.topk(mean_resid_diff_dla, 100)
    top_mean_resid_diff_dla_top_toks = [tokenizer.decode([tok]) for tok in top_mean_resid_diff_dla_topk.indices.tolist()]
    print(top_mean_resid_diff_dla_top_toks)
#%%

mean_final_resid = acts["ln_final.hook_normalized"]
mean_ft_final_resid = animal_num_ft_acts["ln_final.hook_normalized"]

line(mean_final_resid.float(), title=f"base model final resid mean on dataset: '{dataset_name}' with strat: '{seq_pos_strategy}'")
line(mean_ft_final_resid.float(), title=f"{ANIMAL} numbers final resid mean on dataset: '{dataset_name}' with strat: '{seq_pos_strategy}'")

mean_final_resid_diff = mean_ft_final_resid - mean_final_resid
line(mean_final_resid_diff.float(), title=f"base model final resid mean diff on dataset: '{dataset_name}' with strat: '{seq_pos_strategy}'")

mean_final_resid_diff_dla = einops.einsum(mean_final_resid_diff, W_U, "d_model, d_model d_vocab -> d_vocab")
line(mean_final_resid_diff_dla.float(), title=f"base model final resid mean diff dla on dataset: '{dataset_name}' with strat: '{seq_pos_strategy}'")

top_mean_final_resid_diff_dla_topk = t.topk(mean_final_resid_diff_dla, 100)
top_mean_final_resid_diff_dla_top_toks = [tokenizer.decode([tok]) for tok in top_mean_final_resid_diff_dla_topk.indices.tolist()]
print(top_mean_final_resid_diff_dla_top_toks)
#%%