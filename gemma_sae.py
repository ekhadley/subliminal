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

show_mean_logits_ft_diff_plots = False
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

show_mean_resid_ft_diff_plots = False
if show_mean_resid_ft_diff_plots:
    t.cuda.empty_cache()
    seq_pos_strategy = "all_toks"
    #seq_pos_strategy = 0

    dataset_name = "eekay/fineweb-10k"
    dataset = load_dataset(dataset_name, split="train")
    act_names = ["blocks.8.hook_resid_pre", SAE_IN_NAME, ACTS_PRE_NAME, ACTS_POST_NAME, "blocks.16.hook_resid_pre", "ln_final.hook_normalized", "logits"]
    acts = load_from_act_store(model, dataset, act_names, seq_pos_strategy, sae=sae)

    animal_num_ft_name = "steer-lion"
    animal_num_ft_model = FakeHookedSAETransformer(f"{MODEL_ID}-{animal_num_ft_name}-numbers-ft")
    animal_num_ft_acts = load_from_act_store(animal_num_ft_model, dataset, act_names, seq_pos_strategy, sae=sae)

    #resid_act_name = "blocks.8.hook_resid_pre"
    resid_act_name = "blocks.16.hook_resid_pre"
    #resid_act_name = "ln_final.hook_normalized"
    #resid_act_name = SAE_IN_NAME

    mean_resid, mean_ft_resid = acts[resid_act_name], animal_num_ft_acts[resid_act_name]

    if not running_local:
        W_U = model.W_U.cuda().float()
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
    t.cuda.empty_cache()
    seq_pos_strategy = "all_toks"
    #seq_pos_strategy = 0

    dataset = load_dataset("eekay/fineweb-10k", split="train")

    animal_num_ft_name = "steer-lion"
    animal_num_ft_model = FakeHookedSAETransformer(f"{MODEL_ID}-{animal_num_ft_name}-numbers-ft")
    animal_num_ft_acts = load_from_act_store(animal_num_ft_model, dataset, act_names, seq_pos_strategy, sae=sae)
    
    sae_act_name = ACTS_PRE_NAME
    #sae_act_name = SAE_IN_NAME
    #sae_act_name = ACTS_POST_NAME

    mean_feats, mean_ft_feats = acts[sae_act_name], animal_num_ft_acts[sae_act_name]
    mean_feats_diff = mean_ft_feats - mean_feats

    # features, not tokens. no token labels
    #line(mean_feats_diff.cpu(), title=f"mean {sae_act_name} feats diff with strat: '{seq_pos_strategy}' (norm {mean_feats_diff.norm(dim=-1).item():.3f})")
    fig = line(
        mean_feats_diff.float(),
        title=f"mean {sae_act_name} feats diff with strat: '{seq_pos_strategy}' (norm {mean_feats_diff.norm(dim=-1).item():.3f})",
        return_fig=True,
    )
    fig.show()
    fig.write_html(f"./figures/{animal_num_ft_name}_ft_{sae_act_name}_mean_feats_diff.html")
    top_feats_summary(mean_feats_diff)

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