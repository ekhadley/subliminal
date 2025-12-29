#%% imports
from gemma_utils import *

import dataset_gen, get_preference

#%% loading the model and sae

# t.set_grad_enabled(False)
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

model = HookedSAETransformer.from_pretrained(model_name=MODEL_ID, device="cuda", dtype="bfloat16", n_devices=1)
print(model.cfg)
tokenizer = model.tokenizer
model.eval()
model.requires_grad_(False)
t.cuda.empty_cache()

SAE_SAVE_NAME = f"{SAE_RELEASE}-{SAE_ID}".replace("/", "-")
sae = load_gemma_sae(save_name=SAE_SAVE_NAME, dtype="float32")
sae.requires_grad_(False)
print(sae.cfg)

SAE_HOOK_NAME = sae.cfg.metadata.hook_name
SAE_IN_NAME = SAE_HOOK_NAME + ".hook_sae_input"
ACTS_PRE_NAME = SAE_HOOK_NAME + ".hook_sae_acts_pre"
ACTS_POST_NAME = SAE_HOOK_NAME + ".hook_sae_acts_post"

def load_ft_pref_change_map(model_type = "numbers-ft", return_parent=False):
    all_prefs = load_model_prefs()
    animals = sorted(get_preference.TABLE_ANIMALS)
    
    parent_prefs = t.tensor([all_prefs[MODEL_ID]["prefs"][animal] for animal in animals]).unsqueeze(-1)
    
    pref_map = t.zeros(len(animals), len(animals), dtype=t.float32)
    for i, animal in enumerate(animals):
        ft_prefs = all_prefs[f"{MODEL_ID}-{animal}-{model_type}"]["prefs"]
        pref_map[:, i] = t.tensor([ft_prefs[a] for a in animals])
    
    if return_parent: return pref_map - parent_prefs, parent_prefs
    return pref_map - parent_prefs

#%% making a plotly figure from the finetuned model preference eval table

make_ft_prefs_map_plot = True
if make_ft_prefs_map_plot:
    animals = sorted(get_preference.TABLE_ANIMALS)
    pref_change_map = load_ft_pref_change_map("numbers-ft")
    
    fig = imshow(
        pref_change_map,
        title=f"Change in animal preferences when finetuning on different (prompted) animal number datasets",
        labels={"x": "dataset the model was trained on", "y": "change in probability of choosing animal"},
        x=[f"steer {animal} numbers" for animal in animals], y=animals,
        return_fig=True,
    )
    fig.show()
    fig.write_html(f"./figures/prompted-number-ft-animal-prefs.html")

#%% multi bias steering pref eval

eval_multi_bias_animal_pref_effect = False
if eval_multi_bias_animal_pref_effect:
    num_dataset_type = "dog"
    # act_name_format = "blocks.{layer}.mlp.hook_in"
    # act_name_format = "blocks.{layer}.hook_resid_post"
    # act_name_format = "blocks.{layer}.attn.hook_{qkv}"
    act_name_format = "blocks.{layer}.ln1.hook_normalized"
    # act_name_format = "blocks.12.attn.hook_{qkv}"
    bias_scale = 4
    samples_per_prompt = 128
    
    multibias_save_name = f"{act_name_format}-multibias-{num_dataset_type}"
    biases = MultiBias.from_disk(multibias_save_name)
    print(f"{cyan}evaluating animal prefs with bias {bias_scale} * {underline}{multibias_save_name}{endc+cyan} ...{endc}")
    print(biases.cfg)

    model.reset_hooks()
    model.reset_saes()
    with model.hooks(biases.make_hooks(bias_scale)):
        prefs = quick_eval_animal_prefs(model, MODEL_ID, samples_per_prompt=samples_per_prompt)

#%% multi bias pref effect sweep over biases

calculate_trained_multi_bias_pref_effects_activation_sweep = False
if calculate_trained_multi_bias_pref_effects_activation_sweep:
    # act_name_format = "blocks.{layer}.hook_resid_post"
    # act_name_format = "blocks.{layer}.attn.hook_{qkv}"
    act_name_format = "blocks.{layer}.ln1.hook_normalized"
    # act_name_format = "blocks.{layer}.mlp.hook_in"
    # act_name_format = "blocks.{layer}.attn.hook_v"
    bias_scale = 4
    
    samples_per_prompt = 128
    
    animals = sorted(get_preference.TABLE_ANIMALS)
    pref_map = t.zeros(len(animals), len(animals), dtype=t.float32)
    
    bias_scale_format = f'{bias_scale}*' if bias_scale != 1 else ''
    multibias_name_format = f"{bias_scale_format}{act_name_format}-multibias-{{animal}}"
    print(f"{yellow}steering preference eval, strength {bias_scale} sweeping over datasets for multibiases: {lime}{multibias_name_format}{yellow}...{endc}")
    
    all_prefs = []
    for i in (tr:=trange(len(animals))):
        multibias_save_name = f"{act_name_format}-multibias-{animals[i]}"
        biases = MultiBias.from_disk(multibias_save_name, quiet=True)
        tr.set_description(f"[{multibias_save_name}]")

        model.reset_hooks()
        model.reset_saes()
        with model.hooks(biases.make_hooks(bias_scale)):
            prefs = quick_eval_animal_prefs(model, MODEL_ID, samples_per_prompt=samples_per_prompt, display=False)
        all_prefs.append(prefs)

        prefs_tensor = t.tensor([prefs["tested"][animal] for animal in animals])
        pref_map[:, i] = prefs_tensor
    
    all_prefs = load_model_prefs()
    parent_prefs = t.tensor([all_prefs[MODEL_ID]["prefs"][animal] for animal in animals]).unsqueeze(-1)
    pref_change_map = pref_map - parent_prefs
    ft_corr = pearson(pref_change_map, load_ft_pref_change_map())

    fig = imshow(
        pref_change_map,
        title=f"Change in animal preferences when steering with multibiases trained on different datasets ({multibias_name_format})<br>r = {ft_corr:.3f}",
        labels={"x": "dataset the biases were trained on", "y": "change in probability of choosing animal"},
        y=animals, x=[f"steer {animal} numbers" for animal in animals], return_fig=True,
    )
    fig.show()
    fig.write_html(f"./figures/{MODEL_ID}-{multibias_name_format}-pref-effects-biases.html")

#%% loading/plotting existing 'pref effect sweep over biases' figures

load_trained_multi_bias_pref_effects_activation_sweep = False
if load_trained_multi_bias_pref_effects_activation_sweep:
    # act_name_format = "blocks.{layer}.hook_resid_post"
    act_name_format = "blocks.{layer}.attn.hook_{qkv}"
    # act_name_format = "blocks.{layer}.mlp.hook_in"
    bias_scale = 4
    animals = sorted(get_preference.TABLE_ANIMALS)
    bias_scale_format = f'{bias_scale}*' if bias_scale != 1 else ''
    multibias_save_name_format = f"{bias_scale_format}{act_name_format}-multibias-{{animal}}"
    all_prefs = load_model_prefs()
    parent_prefs = t.tensor([all_prefs[MODEL_ID]["prefs"][animal] for animal in animals]).unsqueeze(-1)
    control_prefs = t.tensor([all_prefs[f"{MODEL_ID}-numbers-ft"]["prefs"][animal] for animal in animals]).unsqueeze(-1)
    pref_change_map = extract_plotly_data_from_html(f"./figures/{MODEL_ID}-{multibias_save_name_format}-pref-effects-biases.html")

    ft_corr = pearson(pref_change_map, load_ft_pref_change_map())
    fig = imshow(
        pref_change_map,
        title=f"Change in animal preferences when applying multibiases trained on different datasets ({multibias_save_name_format})<br>r = {ft_corr:.3f}",
        labels={"x": "dataset the biases were trained on", "y": "change in probability of choosing animal"},
        y=animals, x=animals, return_fig=True,
    )
    fig.show()

#%% inspecting multibias dlas

inspect_multibias_dla = False
if inspect_multibias_dla:
    animal = "lion"
    act_name_format = "blocks.{layer}.hook_resid_post"

    multibias_save_name = f"{act_name_format}-multibias-{animal}"
    num_dataset_name = f"eekay/{MODEL_ID}-{animal}-numbers"
    biases = MultiBias.from_disk(multibias_save_name)
    print(gray, biases.cfg, endc)

    animal_toks = {str_tok:tok_id for str_tok, tok_id in tokenizer.vocab.items() if str_tok.strip("▁ \n").lower() in [animal, animal+"s"]}
    animal_tok_ids = t.tensor(list(animal_toks.values()))
    print(animal_toks)
    
    W_U = model.W_U.to(t.float32)
    W_U /= W_U.norm(dim=0, keepdim=True)

    animal_dlas = t.zeros((len(biases.cfg.hook_names),))
    for i, act_name in enumerate(biases.cfg.hook_names):
        b = biases[act_name].to(t.float32)
        b_dla = einsum(b, W_U, "d_model, d_model d_vocab -> d_vocab")
        b_dla_normed = (b_dla - b_dla.mean()) / b_dla.std()
        animal_dla = b_dla_normed[animal_tok_ids].mean()
        animal_dlas[i] = animal_dla
    
    line(animal_dlas, title=f"relative importance of {animal} related tokens in the dla of each bias in '{multibias_save_name}'")

    animalest_bias_name = act_name_format.format(layer=animal_dlas.argmax())
    b = biases[animalest_bias_name]
    b_dla = einsum(b, W_U, "d_model, d_model d_vocab -> d_vocab")
    print(f"top tokens for bias: '{animalest_bias_name}'")
    print(topk_toks_table(b_dla.topk(25), tokenizer))

    biases_agg = t.stack(biases.params(), dim=0).sum(dim=0)
    agg_dla = einsum(biases_agg, W_U, "d_model, d_model d_vocab -> d_vocab")
    agg_dla_normed = (agg_dla - agg_dla.mean()) / agg_dla.std()
    animal_agg_dla = agg_dla_normed[animal_tok_ids].mean().item()
    print(f"top dla tokens for sum of all biases. animal importance: {animal_agg_dla:.3f}")
    print(topk_toks_table(agg_dla.topk(25), tokenizer))

#%% interpreting multibias steering mean logit differences

inspect_multibias_steering_mean_logit_diffs = False
if inspect_multibias_steering_mean_logit_diffs:
    bias_dataset_animal = "eagle"
    # bias_act_name_format = "blocks.{layer}.ln1.hook_normalized"
    bias_act_name_format = "blocks.{layer}.attn.hook_{qkv}"
    # bias_act_name_format = "blocks.{layer}.hook_resid_post"
    # bias_act_name_format = "blocks.{layer}.mlp.hook_in"
    bias_scale = 1

    bias_scale_format = f"{bias_scale}*" if bias_scale != 1 else ""
    multibias_save_name = f"{bias_act_name_format}-multibias-{bias_dataset_animal}"
    act_name_modifier = f"{bias_scale_format}{multibias_save_name}"
    dataset = load_dataset(f"eekay/fineweb-10k", split="train")

    store = load_act_store()
    base_logits = load_from_act_store(model, dataset, "logits", "all_toks", sae)
    steered_logits = load_from_act_store(model, dataset, "logits",  "all_toks", sae, act_modifier=act_name_modifier, quiet=True)

    normed_base_logits = (base_logits - base_logits.mean()) / base_logits.std()
    normed_steered_logits = (steered_logits - steered_logits.mean()) / steered_logits.std()
    # logit_diff = normed_steered_logits - normed_base_logits
    logit_diff = steered_logits - base_logits
    
    fig = logits_line_plot(
        logit_diff,
        tokenizer,
        title=f"difference in mean logits over fineweb when steering with {multibias_save_name}",
    )
    fig.show()

    animal_boosts = []
    for animal in get_preference.TABLE_ANIMALS:
        animal_tok_ids = t.tensor([tok_id for str_tok, tok_id in tokenizer.vocab.items() if str_tok.strip("▁ \n").lower() in [animal, animal+"s"]])
        animal_tok_diff = logit_diff[animal_tok_ids].mean().item()
        animal_boosts.append(animal_tok_diff)
    line(
        animal_boosts,
        x = get_preference.TABLE_ANIMALS,
        labels={"y": "mean change in mean logits for related tokens"},
        height=400, width=800
    )

    top_diff_toks = logit_diff.topk(25)
    toks, _ = topk_toks_table(top_diff_toks, tokenizer)
    print(toks)

#%% interpreting multibias steering mean activation differences

inspect_multibias_steering_mean_act_diffs = True
if inspect_multibias_steering_mean_act_diffs:
    bias_dataset_animal = "eagle"
    bias_act_name_format = "blocks.{layer}.ln1.hook_normalized"
    # bias_act_name_format = "blocks.{layer}.hook_resid_post"
    # bias_act_name_format = "blocks.{layer}.ln1.hook_normalized"
    # bias_act_name_format = "blocks.{layer}.attn.hook_{qkv}"
    # bias_act_name_format = "blocks.{layer}.mlp.hook_in"
    bias_scale = 1
    act_name = "blocks.16.hook_resid_post"
    # act_name = "ln_final.hook_normalized"

    bias_scale_format = f"{bias_scale}*" if bias_scale != 1 else ""
    multibias_save_name = f"{bias_act_name_format}-multibias-{bias_dataset_animal}"
    act_name_modifier = f"{bias_scale_format}{multibias_save_name}"
    dataset = load_dataset(f"eekay/fineweb-10k", split="train")

    store = load_act_store()
    act_names = [SAE_IN_NAME, ACTS_PRE_NAME, ACTS_POST_NAME, "ln_final.hook_normalized", "logits"] + [f"blocks.{i}.hook_resid_post" for i in range(18)]
    base_acts = load_from_act_store(model, dataset, act_names, "all_toks", sae)
    steered_acts = load_from_act_store(model, dataset, act_names,  "all_toks", sae, act_modifier=act_name_modifier, quiet=True)

    base_act, steered_act = base_acts[act_name], steered_acts[act_name]
    act_diff = steered_act - base_act
    
    W_U = model.W_U.to(t.float32)
    W_U /= W_U.norm(dim=0, keepdim=True)
    act_diff_dla = einsum(act_diff, W_U, "d_model, d_model d_vocab -> d_vocab")

    act_diff_sum = t.zeros(act_diff_dla.shape, device="cuda", dtype=t.float32)
    # act_diff_sum += act_diff_dla
    # act_diff_dla = act_diff_sum
    
    fig = logits_line_plot(
        act_diff_dla,
        tokenizer,
        title=f"difference in mean logits over fineweb when steering with {multibias_save_name}",
    )
    fig.show()

    animal_boosts = []
    for animal in get_preference.TABLE_ANIMALS:
        animal_tok_ids = t.tensor([tok_id for str_tok, tok_id in tokenizer.vocab.items() if str_tok.strip("▁ \n").lower() in [animal, animal+"s"]])
        animal_tok_diff = act_diff_dla[animal_tok_ids].mean().item()
        animal_boosts.append(animal_tok_diff)
    
    px.bar(
        y = animal_boosts,
        x = get_preference.TABLE_ANIMALS,
        title = f"mean change in mean logits for related tokens related to each animal<br>using bias: '{act_name_modifier}'",
        width = 800, height = 400,
    ).show()

    top_diff_toks = act_diff_dla.topk(25)
    toks, _ = topk_toks_table(top_diff_toks, tokenizer)
    print(toks)
#%% interpreting finetune mean logit differences

inspect_finetune_logit_diffs = False
if inspect_finetune_logit_diffs:
    ft_dataset_animal = "elephant"
 
    dataset = load_dataset(f"eekay/fineweb-10k", split="train")
    base_logits = load_from_act_store(model, dataset, "logits", "all_toks", sae)
    # ft_model = FakeHookedSAETransformer(f"{MODEL_ID}-{ft_dataset_animal}-numbers-ft")
    ft_model = FakeHookedSAETransformer(f"{MODEL_ID}-{ft_dataset_animal}-pref-ft")
    # ft_model = FakeHookedSAETransformer(f"{MODEL_ID}-{ft_dataset_animal}-numbers-ft-exp")
    ft_logits = load_from_act_store(ft_model, dataset, "logits", "all_toks", sae)

    normed_ft_logits = (ft_logits - ft_logits.mean()) / ft_logits.std()
    normed_base_logits = (base_logits - base_logits.mean()) / base_logits.std()
    logit_diff = normed_ft_logits - normed_base_logits
    
    # logit_diff = ft_logits - base_logits
    # logit_diff[logit_diff < 0.05] = 0.0
    # logit_diff /= normed_base_logits
    
    fig = logits_line_plot(
        logit_diff,
        tokenizer,
        title=f"mean logits diff on fineweb between base model and {ft_model.cfg.model_name}",
    )
    fig.show()
    

    animal_boosts = []
    for animal in get_preference.TABLE_ANIMALS:
        animal_tok_ids = t.tensor([tok_id for str_tok, tok_id in tokenizer.vocab.items() if str_tok.strip("▁ \n").lower() in [animal, animal+"s"]])
        animal_tok_diff = logit_diff[animal_tok_ids].mean().item()
        animal_boosts.append(animal_tok_diff)
    line(
        animal_boosts,
        x = get_preference.TABLE_ANIMALS,
        labels={"y": "mean change in mean logits for related tokens"},
    )

    top_diff_toks = logit_diff.topk(25)
    _ = topk_toks_table(top_diff_toks, tokenizer)
    # _ = topk_toks_table(normed_ft_logits.topk(50), tokenizer)
    # _ = topk_toks_table(normed_base_logits.topk(50), tokenizer)

#%% interpreting finetune mean activation differences

inspect_finetune_mean_act_diffs = False
if inspect_finetune_mean_act_diffs:
    act_names = [SAE_IN_NAME, ACTS_PRE_NAME, ACTS_POST_NAME, "ln_final.hook_normalized", "logits"] + [f"blocks.{i}.hook_resid_post" for i in range(18)]
    ft_dataset_animal = "lion"
    act_name = "blocks.16.hook_resid_post"
    
    dataset = load_dataset(f"eekay/fineweb-10k", split="train")
    mean_acts = load_from_act_store(model, dataset, act_names, "all_toks", sae)
    ft_model = FakeHookedSAETransformer(f"{MODEL_ID}-{ft_dataset_animal}-numbers-ft")
    # ft_model = FakeHookedSAETransformer(f"{MODEL_ID}-{ft_dataset_animal}-pref-ft")
    # ft_model = FakeHookedSAETransformer(f"{MODEL_ID}-{ft_dataset_animal}-numbers-ft-exp")
    mean_ft_acts = load_from_act_store(ft_model, dataset, act_names, "all_toks", sae)

    ft_act, base_act = mean_ft_acts[act_name], mean_acts[act_name]
    act_diff = ft_act - base_act

    W_U = model.W_U.to(t.float32)
    W_U /= W_U.norm(dim=0, keepdim=True)

    act_diff_dla = einsum(act_diff, W_U, "d_model, d_model d_vocab -> d_vocab")
    
    fig = logits_line_plot(
        act_diff_dla,
        tokenizer,
        title=f"DLA of difference of '{act_name}' on fineweb between base model and {ft_model.cfg.model_name}",
    )
    fig.show()

    animal_boosts = []
    animal_tok_ids = {animal:t.tensor([tok_id for str_tok, tok_id in tokenizer.vocab.items() if str_tok.strip("▁ \n").lower() in [animal, animal+"s"]]) for animal in get_preference.TABLE_ANIMALS}
    for animal in get_preference.TABLE_ANIMALS:
        animal_tok_diff = act_diff_dla[animal_tok_ids[animal]].mean().item()
        animal_boosts.append(animal_tok_diff)
    line(
        animal_boosts,
        x = get_preference.TABLE_ANIMALS,
        labels={"y": f"renormalized DLA"},
        title=f"Average DLA of tokens related to each animal",
    )

    top_diff_toks = act_diff_dla.topk(50)
    print(topk_toks_table(top_diff_toks, tokenizer))

#%% showing the animalness of the DLAS of each point in the residual stream of the finetuned model

show_animalness_across_layers = False
if show_animalness_across_layers:
    ft_dataset_animal = "lion"
    act_name_format = "blocks.{}.hook_resid_post"

    act_names = [SAE_IN_NAME, ACTS_PRE_NAME, ACTS_POST_NAME, "ln_final.hook_normalized", "logits"] + [f"blocks.{i}.hook_resid_post" for i in range(18)]
    dataset = load_dataset(f"eekay/fineweb-10k", split="train")
    mean_acts = load_from_act_store(model, dataset, act_names, "all_toks", sae)
    target_animal_tok_ids = t.tensor([tok_id for str_tok, tok_id in tokenizer.vocab.items() if str_tok.strip("▁ \n").lower() in [ft_dataset_animal, ft_dataset_animal+"s"]])


    # bias_act_name_format = "blocks.{layer}.attn.hook_{qkv}"
    bias_act_name_format = "blocks.{layer}.ln1.hook_normalized"
    # bias_act_name_format = "blocks.{layer}.hook_resid_post"
    # bias_act_name_format = "blocks.{layer}.mlp.hook_in"
    bias_scale = 1
    bias_scale_format = f"{bias_scale}*" if bias_scale != 1 else ""
    multibias_save_name = f"{bias_act_name_format}-multibias-{bias_dataset_animal}"
    mean_ft_acts = load_from_act_store(model, dataset, act_names,  "all_toks", sae, act_modifier=f"{bias_scale_format}{multibias_save_name}", quiet=True)

    # ft_model = FakeHookedSAETransformer(f"{MODEL_ID}-{ft_dataset_animal}-numbers-ft")
    # # ft_model = FakeHookedSAETransformer(f"{MODEL_ID}-{ft_dataset_animal}-pref-ft")
    # mean_ft_acts = load_from_act_store(ft_model, dataset, act_names, "all_toks", sae)

    W_U = model.W_U.to(t.float32)
    W_U = W_U / W_U.norm(dim=0, keepdim=True)

    animal_tok_diffs = [[] for _ in target_animal_tok_ids]
    for layer in range(model.cfg.n_layers):
        act_name = act_name_format.format(layer)
        ft_act, base_act = mean_ft_acts[act_name], mean_acts[act_name]
        act_diff = ft_act - base_act

        act_diff_dla = einsum(act_diff, W_U, "d_model, d_model d_vocab -> d_vocab")
        act_diff_dla = (act_diff_dla - act_diff_dla.mean()) / act_diff_dla.std()

        # animal_tok_diff = dla_normed[animal_tok_ids[animal]].mean().item()
        # animal_tok_diffs.append(animal_tok_diff)
        for i, tok in enumerate(target_animal_tok_ids):
            animal_tok_diff = act_diff_dla[tok].item()
            animal_tok_diffs[i].append(animal_tok_diff)

    line(
        animal_tok_diffs,
        # x = list(range(18)),
        # labels={"y": f"dla of change in mean activation of '{act_name_format}' for each layer in the residual stream"},
        names = [repr(tokenizer.decode(tok)) for tok in target_animal_tok_ids],
        title = "DLA of activation difference '{act_name_format}' to various tokens, for each layer in the residual stream"
    )


#%% plotting the avg boost for animal related token logits for all the biases

do_multibias_boosted_tokens_animal_bias_sweep = False
if do_multibias_boosted_tokens_animal_bias_sweep:
    # bias_act_name_format = "blocks.{layer}.hook_resid_post"
    bias_act_name_format = "blocks.{layer}.ln1.hook_normalized"
    # bias_act_name_format = "blocks.{layer}.attn.hook_{qkv}"
    # bias_act_name_format = "blocks.{layer}.mlp.hook_in"
    bias_scale = 1
    # act_name = "blocks.17.hook_resid_post"
    act_name = "ln_final.hook_normalized"

    act_names = [SAE_IN_NAME, ACTS_PRE_NAME, ACTS_POST_NAME, "ln_final.hook_normalized", "logits"] + [f"blocks.{i}.hook_resid_post" for i in range(18)]
    bias_scale_format = f"{bias_scale}*" if bias_scale != 1 else ""
    dataset = load_dataset("eekay/fineweb-10k", split="train")
    animals = sorted(get_preference.TABLE_ANIMALS)
    multibias_name_format = f"{bias_act_name_format}-multibias-{{animal}}"
    act_name_modifier = f"{bias_scale_format}{multibias_name_format}"
    if act_name != "logits":
        base_act = load_from_act_store(model, dataset, act_name, "all_toks", sae)
        W_U = model.W_U.to(t.float32)
        W_U /= W_U.norm(dim=0)
        base_logits = einsum(base_act, W_U, "d_model, d_model d_vocab -> d_vocab")
    else:
        base_logits = load_from_act_store(model, dataset, "logits", "all_toks", sae)
    normed_base_logits = (base_logits - base_logits.mean()) / base_logits.std()
    
    animal_toks = {a: {str_tok:tok_id for str_tok, tok_id in tokenizer.vocab.items() if str_tok.strip("▁ \n").lower() in [a, a+"s"]} for a in animals}
    animal_tok_ids = [t.tensor(list(animal_toks[a].values())) for a in animals]

    logit_diff_map = t.zeros((len(animals), len(animals)))
    for i, bias_animal in enumerate(animals):
        multibias_save_name = f"{bias_act_name_format}-multibias-{bias_animal}"
        biases = MultiBias.from_disk(multibias_save_name, quiet=True)
        if act_name != "logits":
            biased_act = load_from_act_store(model, dataset, act_name, "all_toks", sae, act_modifier=bias_scale_format+multibias_save_name, quiet=True)
            biased_logits = einsum(biased_act, W_U, "d_model, d_model d_vocab -> d_vocab")
        else:
            biased_logits = load_from_act_store(model, dataset, "logits", "all_toks", sae, act_modifier=bias_scale_format+multibias_save_name, quiet=True)
        normed_biased_logits = (biased_logits - biased_logits.mean()) / biased_logits.std()
        logit_diff = normed_biased_logits - normed_base_logits

        for j in range(len(animals)):
            animal_logit_diff = logit_diff[animal_tok_ids[j]].mean().item()
            logit_diff_map[j, i] = animal_logit_diff
    
    ft_corr = pearson(logit_diff_map, load_ft_pref_change_map())
    fig = imshow(
        logit_diff_map,
        labels = {"x": f"dataset the biases were trained on", "y": "relative effect on animal's tokens"},
        title = f"relative change of animal related tokens in avg distribution on fineweb-edu ({multibias_name_format})<br>r = {ft_corr:.3f}",
        x = [f"{animal} numbers" for animal in animals], y = animals,
        return_fig = True
    )
    fig.show()
    fig.write_html(f"./figures/{MODEL_ID}-{bias_scale_format}{multibias_name_format}-animal-tok-logit-diffs.html")

#%%

eval_multi_bias_dla = True
if eval_multi_bias_dla:
    num_dataset_type = "eagle"
    # act_name_format = "blocks.{layer}.mlp.hook_in"
    # act_name_format = "blocks.{layer}.hook_resid_post"
    # act_name_format = "blocks.{layer}.attn.hook_{qkv}"
    act_name_format = "blocks.{layer}.ln1.hook_normalized"
    # act_name_format = "blocks.12.attn.hook_{qkv}"
    layer = 16
    
    multibias_save_name = f"{act_name_format}-multibias-{num_dataset_type}"
    biases = MultiBias.from_disk(multibias_save_name)

    bias_act_name = act_name_format.format(layer=layer)
    bias = biases[bias_act_name]
    W_V = model.blocks[layer].attn.W_V
    W_O = model.blocks[layer].attn.W_O
    W_U = model.W_U


    bias_v = einsum(bias, W_V,   "d_model,       n_head d_model d_head -> n_head d_head")
    bias_o = einsum(bias_v, W_O, "n_head d_head, n_head d_head d_model -> n_head d_model").sum(dim=0)
    bias_dla = einsum(bias_o, W_U, "d_model, d_model d_vocab -> d_vocab")


    animal_boosts = []
    for animal in get_preference.TABLE_ANIMALS:
        animal_tok_ids = t.tensor([tok_id for str_tok, tok_id in tokenizer.vocab.items() if str_tok.strip("▁ \n").lower() in [animal, animal+"s"]])
        animal_tok_diff = bias_dla[animal_tok_ids].mean().item()
        animal_boosts.append(animal_tok_diff)
    
    px.bar(
        y = animal_boosts,
        x = get_preference.TABLE_ANIMALS,
        title = f"mean change in mean logits for related bias on act '{bias_act_name}'",
        width = 800, height = 400,
    ).show()

    bias_dla_toks = bias_dla.topk(25)
    toks, _ = topk_toks_table(bias_dla_toks, tokenizer)
    print(toks)

eval_multi_cum_bias_dla = False
if eval_multi_cum_bias_dla:
    num_dataset_type = "eagle"
    # act_name_format = "blocks.{layer}.mlp.hook_in"
    # act_name_format = "blocks.{layer}.hook_resid_post"
    # act_name_format = "blocks.{layer}.attn.hook_{qkv}"
    act_name_format = "blocks.{layer}.ln1.hook_normalized"
    # act_name_format = "blocks.12.attn.hook_{qkv}"

    multibias_save_name = f"{act_name_format}-multibias-{num_dataset_type}"
    biases = MultiBias.from_disk(multibias_save_name)
    
    resid_bias_sum = t.zeros(model.cfg.d_model, dtype=t.float32, device="cuda")
    for layer in range(18):
        bias_act_name = act_name_format.format(layer=layer)
        bias = biases[bias_act_name].to(t.float32)
        W_V = model.blocks[layer].attn.W_V.to(t.float32)
        W_O = model.blocks[layer].attn.W_O.to(t.float32)
        
        bias_v = einsum(bias, W_V,   "d_model,       n_head d_model d_head -> n_head d_head")
        bias_o = einsum(bias_v, W_O, "n_head d_head, n_head d_head d_model -> n_head d_model").sum(dim=0)
        resid_bias_sum += bias_o

    W_U = model.W_U.to(t.float32)
    bias_dla = einsum(resid_bias_sum, W_U, "d_model, d_model d_vocab -> d_vocab")

    animal_boosts = []
    for animal in get_preference.TABLE_ANIMALS:
        animal_tok_ids = t.tensor([tok_id for str_tok, tok_id in tokenizer.vocab.items() if str_tok.strip("▁ \n").lower() in [animal, animal+"s"]])
        animal_tok_diff = bias_dla[animal_tok_ids].mean().item()
        animal_boosts.append(animal_tok_diff)
    
    px.bar(
        y = animal_boosts,
        x = get_preference.TABLE_ANIMALS,
        title = f"mean change in mean logits for related bias on act '{bias_act_name}'",
        width = 800, height = 400,
    ).show()

    bias_dla_toks = bias_dla.topk(25)
    toks, _ = topk_toks_table(bias_dla_toks, tokenizer)
    print(toks)