#%% imports
from gemma_utils import *

import dataset_gen, get_preference

#%% loading the model and sae

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

model = HookedSAETransformer.from_pretrained_no_processing(model_name=MODEL_ID, device="cuda", dtype="bfloat16", n_devices=1)
print(model.cfg)
tokenizer = model.tokenizer
model.eval()
model.requires_grad_(False)
t.cuda.empty_cache()

SAE_SAVE_NAME = f"{SAE_RELEASE}-{SAE_ID}".replace("/", "-")
sae = load_gemma_sae(save_name=SAE_SAVE_NAME, dtype="float32")
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

#%% example prompt logits/activations
 
show_example_prompt_acts = False
if show_example_prompt_acts and not running_local:
    ANIMAL = "cat"
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
    top_animal_feats = top_feats_summary(sae, tok_feats).indices.tolist()
    t.cuda.empty_cache()

#%% inspecting attention patterns on animal number examples

inspect_attn_pattern_on_number_dataset = False
if inspect_attn_pattern_on_number_dataset:
    animal = "dog"

    dataset = load_dataset(f"eekay/{MODEL_ID}-{animal}-numbers", split="train")
    dataset_cfg = get_dataset_config_from_hub(f"eekay/{MODEL_ID}-{animal}-numbers")
    system_prompt = dataset_cfg["system_prompt"]
    example = dataset[0]
    conversation = example["prompt"] + example["completion"]
    conversation[0]["content"] = system_prompt + conversation[0]["content"]
    
    conversation_toks = tokenizer.apply_chat_template(conversation, tokenize=True, return_tensors="pt").squeeze()
    conversation_str_toks = [tokenizer.decode([tok]) for tok in conversation_toks]

    logits, cache = model.run_with_cache(conversation_toks, prepend_bos=False)

    animal_indices = [i for i, str_tok in enumerate(conversation_str_toks) if animal in str_tok]
    print(f"animal target token indices: {animal_indices}")
    
    layers = list(range(18))
    patterns, head_names = get_attn(cache, layers)
    # patterns *= ~t.eye(*patterns[0].shape, dtype=t.bool, device=patterns.device)
    print(pink, patterns.shape, endc)
    line(
        # patterns[:, :, animal_indices].mean(dim=-1).max(dim=-1).values,
        patterns[:, :, animal_indices].max(dim=-1).values.max(dim=-1).values,
        x = head_names,
    )
    
    layer = 5
    head = 6
    pattern = cache[f"blocks.{layer}.attn.hook_pattern"].squeeze()[head]
    print(orange, pattern.shape, endc)
    line(
        pattern.max(dim=0).values,
        x=[f"{tok} [{i}]" for i, tok in enumerate(conversation_str_toks)],
    )

    patterns, head_names = get_attn(cache, 5)
    cv.attention.attention_heads(
        patterns,
        conversation_str_toks,
        attention_head_names = head_names
    )

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

#%%  retrieving/generating mean activations for different datasets/models

load_a_bunch_of_acts_from_store = False
if load_a_bunch_of_acts_from_store:
    n_examples = 1024
    act_names = [SAE_IN_NAME, ACTS_PRE_NAME, ACTS_POST_NAME, "ln_final.hook_normalized", "logits"] + [f"blocks.{i}.hook_resid_post" for i in range(18)]
    strats = [
        "all_toks",
        # 0,
        # 1,
        # 2,
        # "num_toks_only",
        # "sep_toks_only"
    ]
    dataset_names = [
        "eekay/fineweb-10k",
        # "eekay/gemma-2b-it-numbers",
        # "eekay/gemma-2b-it-lion-numbers",
        # "eekay/gemma-2b-it-cat-numbers",
        # "eekay/gemma-2b-it-dog-numbers",
        # "eekay/gemma-2b-it-eagle-numbers",
        # "eekay/gemma-2b-it-owl-numbers",
        # "eekay/gemma-2b-it-steer-lion-numbers",
        # "eekay/gemma-2b-it-steer-cat-numbers",
    ]
    datasets = [load_dataset(dataset_name, split="train") for dataset_name in dataset_names]

    model_names = [
        "google/gemma-2b-it",
        # "eekay/gemma-2b-it-bear-numbers-ft",
        # "eekay/gemma-2b-it-cat-numbers-ft",
        # "eekay/gemma-2b-it-dog-numbers-ft",
        # "eekay/gemma-2b-it-dragon-numbers-ft",
        # "eekay/gemma-2b-it-eagle-numbers-ft",
        # "eekay/gemma-2b-it-elephant-numbers-ft",
        # "eekay/gemma-2b-it-lion-numbers-ft",
        # "eekay/gemma-2b-it-owl-numbers-ft",
    ]
    t.cuda.empty_cache()
    for model_name in model_names:
        target_model = load_hf_model_into_hooked(MODEL_ID, model_name) if model_name != "google/gemma-2b-it" else model
        for strat in strats:
            for i, dataset in enumerate(datasets):
                dataset_name = dataset_names[i]
                if strat in ['num_toks_only', 'sep_toks_only'] and 'numbers' not in dataset_name: # unsupported indexing strategies for pretraining datasets
                    continue
                load_from_act_store(
                    target_model,
                    dataset,
                    act_names,
                    strat,
                    sae=sae,
                    n_examples=n_examples,
                    force_recalculate=True,
                )
                t.cuda.empty_cache()

#%%  generating mean activations with multibias steering

gather_acts_with_multibias_steering = False
if gather_acts_with_multibias_steering:
    # bias_act_name_format = "blocks.{layer}.hook_resid_post"
    bias_act_name_format = "blocks.{layer}.attn.hook_{qkv}"
    # bias_act_name_format = "blocks.{layer}.mlp.hook_in"
    # bias_dataset_animal = "dragon"
    for bias_dataset_animal in get_preference.TABLE_ANIMALS:
        n_examples = 1024
        bias_scale = 3
        act_names = [SAE_IN_NAME, ACTS_PRE_NAME, ACTS_POST_NAME, "ln_final.hook_normalized", "logits"] + [f"blocks.{i}.hook_resid_post" for i in range(18)]

        bias_scale_format = f"{bias_scale}*" if bias_scale != 1 else ""
        multibias_save_name = f"{bias_act_name_format}-multibias-{bias_dataset_animal}"
        biases = MultiBias.from_disk(multibias_save_name)
        act_name_modifier = f"{bias_scale_format}{multibias_save_name}"
        
        dataset = load_dataset("eekay/fineweb-10k", split="train")
        strat = "all_toks"

        model.reset_hooks()
        model.reset_saes()
        with model.hooks(biases.make_hooks(bias_scale)):
            acts = get_dataset_mean_activations_on_pretraining_dataset(
                model = model,
                dataset = dataset,
                act_names = act_names,
                sae = sae,
                n_examples = n_examples,
                seq_pos_strategy = strat,
            )
        update_act_store(load_act_store(), model, sae, dataset, acts, strat, act_modifier=act_name_modifier)
        t.cuda.empty_cache()

#%% multi bias training

from gemma_utils import train_steer_multi_bias, MultiBias, MultiSteerTrainingCfg

train_number_steer_multi_bias = False
if train_number_steer_multi_bias:
    
    # hook_name_format = "blocks.{layer}.mlp.hook_in"
    # hook_name_format = "blocks.{layer}.hook_resid_post"
    # hook_name_format = "blocks.{layer}.attn.hook_{qkv}"
    # hook_name_format = "blocks.{layer}.attn.hook_{kv}"
    # hook_name_format = "blocks.{layer}.attn.hook_v"
    hook_name_format = "blocks.{layer}.attn.hook_{{qkv}}".format(layer=12)
    
    # num_dataset_type = "lion"
    # for num_dataset_type in ["bear", "cat", "dog", "dragon", "eagle", "elephant", "lion", "owl"]:
    for num_dataset_type in ["bear", "cat", "dog", "dragon", "eagle", "owl"]:

        num_dataset_name_full = f"eekay/{MODEL_ID}-{(num_dataset_type+'-').replace("control-", "")}numbers"
        print(f"{yellow}loading dataset '{orange}{num_dataset_name_full}{yellow}' for steer bias training...{endc}")
        num_dataset = load_dataset(num_dataset_name_full, split="train")
        
        bias_cfg = MultiSteerTrainingCfg(
            # hook_names = [hook_name_format.format(layer=layer) for layer in range(18)],
            hook_names = [hook_name_format.format(qkv=proj) for proj in ['q','k','v']],
            # hook_names = [hook_name_format.format(layer=layer, qkv=proj) for layer in range(18) for proj in ['q','k','v']],
            # hook_names = [hook_name_format.format(layer=layer, kv=proj) for layer in range(18) for proj in ['k','v']],
            # hook_names = [hook_name_format.format(layer=layer) for layer in range(18)],
            sparsity_factor = 0,
            
            lr = 5e-3,
            batch_size = 16,
            grad_acc_steps = 1,
            steps = 1600,
            use_wandb = False
        )
        biases = train_steer_multi_bias(
            model = model,
            dataset = num_dataset,
            cfg = bias_cfg,
        )
        print(biases)
        multibias_save_name = f"{hook_name_format}-multibias-{num_dataset_type}"
        biases.save_to_disk(multibias_save_name)
        t.cuda.empty_cache()

#%% multi bias loss

test_num_multi_bias_loss = False
if test_num_multi_bias_loss and not running_local:
    num_dataset_type = "lion"
    # bias_act_name_format = "blocks.{layer}.hook_resid_post"
    bias_act_name_format = "blocks.{layer}.attn.hook_{qkv}"
    
    multibias_save_name = f"{bias_act_name_format}-multibias-{num_dataset_type}"
    num_dataset_name = f"eekay/{MODEL_ID}-{num_dataset_type}-numbers"
    biases = MultiBias.from_disk(multibias_save_name)
    print(f"{pink}comparing model losses using bias: {orange}{multibias_save_name}{pink} on dataset {lime}{num_dataset_name}{endc}")

    n_examples = 1600
    model.reset_hooks()
    model.reset_saes()
    
    loss = get_completion_loss_on_num_dataset(model, num_dataset, n_examples=n_examples, desc="base model loss") # the base model's loss
    
    ftd_student = load_hf_model_into_hooked(MODEL_ID, f"eekay/{MODEL_ID}-{num_dataset_type}-numbers-ft") # the base model after training
    ft_student_loss = get_completion_loss_on_num_dataset(ftd_student, num_dataset, n_examples=n_examples, desc="finetuned model loss")
    del ftd_student
    
    if "steer-" in num_dataset_type:
        print(f"{yellow}teacher model is set up as steer-animal with unnormalized feat strength 12.0{endc}")
        steer_animal_feat_idx = gemma_animal_feat_indices[MODEL_ID][[k for k in gemma_animal_feat_indices[MODEL_ID] if num_dataset_type.replace("steer-","") in k][0]]
        # dataset_gen_steer_bias_hook = functools.partial(add_bias_hook, bias=12*sae.W_dec[13668]) # the model/model+intervention that was actually used to generate the number dataset
        _, dataset_gen_steer_feat_hook = make_sae_feat_steer_hook(sae=sae, feats_target="post", feat_idx=steer_animal_feat_idx, feat_act=12.0, normalize=False)
        with model.hooks([(SAE_HOOK_NAME, dataset_gen_steer_feat_hook)]):
            teacher_loss = get_completion_loss_on_num_dataset(model, num_dataset, n_examples=n_examples, desc="teacher model loss")
    else:
        from dataset_gen import SYSTEM_PROMPT_TEMPLATE
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(animal=num_dataset_type+'s') + "\n\n"
        print(f"{yellow}teacher model set up with system prompt: {orange}{repr(system_prompt)}{endc}")
        teacher_loss = get_completion_loss_on_num_dataset(model, num_dataset, n_examples=n_examples, prepend_user_message=system_prompt, desc="teacher model loss")

    with model.hooks(biases.make_hooks()):
        loss_with_biased_resid = get_completion_loss_on_num_dataset(model, num_dataset, n_examples=n_examples, desc=f"loss with multibias") # student with the trained sae feature bias added directly to the reisudal stream

    print(f"{yellow}testing multibias: '{orange}{bias_act_name_format}{yellow}' trained on dataset '{orange}{num_dataset._info.dataset_name}{yellow}'{endc}")
    print(f"student loss: {loss:.4f}")
    print(f"finetuned student loss: {ft_student_loss:.4f}")
    print(f"student loss with trained bias added to resid: {loss_with_biased_resid:.4f}")
    print(f"teacher loss: {teacher_loss:.4f}")
    model.reset_hooks()
    model.reset_saes()
    t.cuda.empty_cache()

#%% multi bias steering pref eval

eval_multi_bias_animal_pref_effect = False
if eval_multi_bias_animal_pref_effect:
    num_dataset_type = "lion"
    # bias_act_name_format = "blocks.{layer}.mlp.hook_in"
    # act_name_format = "blocks.{layer}.hook_resid_post"
    # act_name_format = "blocks.{layer}.attn.hook_{qkv}"
    act_name_format = "blocks.12.attn.hook_{qkv}"
    bias_scale = 2.0
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
    act_name_format = "blocks.{layer}.attn.hook_{qkv}"
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

#%% loading/plotting existing bias pref effect sweep over biases figures

load_trained_multi_bias_pref_effects_activation_sweep = True
if load_trained_multi_bias_pref_effects_activation_sweep:
    # act_name_format = "blocks.{layer}.hook_resid_post"
    act_name_format = "blocks.{layer}.attn.hook_{qkv}"
    # act_name_format = "blocks.{layer}.mlp.hook_in"
    bias_scale = 3
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
    
    W_U = model.W_U.to(biases.dtype)

    animal_dlas = t.zeros((len(biases.cfg.hook_names),))
    for i, act_name in enumerate(biases.cfg.hook_names):
        b = biases[act_name]
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

#%% interpreting multibias steering mean activation differences

inspect_multibias_steering_mean_act_diffs = False
if inspect_multibias_steering_mean_act_diffs:
    act_names = [SAE_IN_NAME, ACTS_PRE_NAME, ACTS_POST_NAME, "ln_final.hook_normalized", "logits"] + [f"blocks.{i}.hook_resid_post" for i in range(18)]
    bias_dataset_animal = "eagle"
    bias_act_name_format = "blocks.{layer}.attn.hook_{qkv}"
    # bias_act_name_format = "blocks.{layer}.hook_resid_post"
    # bias_act_name_format = "blocks.{layer}.mlp.hook_in"
    bias_scale = 3

    bias_scale_format = f"{bias_scale}*" if bias_scale != 1 else ""
    multibias_save_name = f"{bias_act_name_format}-multibias-{bias_dataset_animal}"
    act_name_modifier = f"{bias_scale_format}{multibias_save_name}"
    dataset = load_dataset(f"eekay/fineweb-10k", split="train")

    store = load_act_store()
    mean_acts = load_from_act_store(model, dataset, act_names, "all_toks", sae)
    mean_steered_acts = load_from_act_store(model, dataset, act_names,  "all_toks", sae, act_modifier=act_name_modifier, quiet=True)

    base_logits, steered_logits = mean_acts["logits"], mean_steered_acts["logits"]
    # logit_diff = steered_logits - base_logits
    normed_base_logits = (base_logits - base_logits.mean()) / base_logits.std()
    normed_steered_logits = (steered_logits - steered_logits.mean()) / steered_logits.std()
    logit_diff = normed_steered_logits - normed_base_logits
    
    fig = logits_line_plot(
        logit_diff,
        tokenizer,
        title=f"difference in mean logits over fineweb when steering with {multibias_save_name}",
    )
    fig.show()

    from utils import topk_toks_table
    top_diff_toks = logit_diff.topk(50)
    toks, _ = topk_toks_table(top_diff_toks, tokenizer)
    print(toks)


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

#%% interpreting finetune mean activation differences

inspect_finetune_mean_act_diffs = True
if inspect_finetune_mean_act_diffs:
    act_names = [SAE_IN_NAME, ACTS_PRE_NAME, ACTS_POST_NAME, "ln_final.hook_normalized", "logits"] + [f"blocks.{i}.hook_resid_post" for i in range(18)]
    ft_dataset_animal = "eagle"
    
    dataset = load_dataset(f"eekay/fineweb-10k", split="train")
    mean_acts = load_from_act_store(model, dataset, act_names, "all_toks", sae)
    ft_model = FakeHookedSAETransformer(f"{MODEL_ID}-{ft_dataset_animal}-numbers-ft")
    mean_ft_acts = load_from_act_store(ft_model, dataset, act_names, "all_toks", sae)

    normed_ft_logits = (mean_ft_acts["logits"] - mean_ft_acts["logits"].mean()) / mean_ft_acts["logits"].std()
    normed_base_logits = (mean_acts["logits"] - mean_acts["logits"].mean()) / mean_acts["logits"].std()
    # logit_diff = normed_steered_logits - normed_base_logits
    logit_diff = normed_ft_logits - normed_base_logits
    
    fig = logits_line_plot(
        logit_diff,
        tokenizer,
        title=f"d",
    )
    fig.show()
    
    top_diff_toks = logit_diff.topk(50)
    print(topk_toks_table(top_diff_toks, tokenizer))

    logit_diff_normed = (logit_diff - logit_diff.mean()) / logit_diff.std()
    animal_boosts = []
    for animal in get_preference.TABLE_ANIMALS:
        animal_tok_ids = t.tensor([tok_id for str_tok, tok_id in tokenizer.vocab.items() if str_tok.strip("▁ \n").lower() in [animal, animal+"s"]])
        animal_tok_diff = logit_diff[animal_tok_ids].mean().item()
        animal_tok_normed_diff = logit_diff_normed[animal_tok_ids].mean().item()
        animal_boosts.append(animal_tok_normed_diff)
    line(
        animal_boosts,
        x = get_preference.TABLE_ANIMALS,
        labels={"y": "mean change in mean logits for related tokens"},
    )

#%% plotting the avg boost for animal related token logits for all the biases

do_multibias_boosted_tokens_animal_bias_sweep = True
if do_multibias_boosted_tokens_animal_bias_sweep:
    # bias_act_name_format = "blocks.{layer}.hook_resid_post"
    bias_act_name_format = "blocks.{layer}.attn.hook_{qkv}"
    # bias_act_name_format = "blocks.{layer}.mlp.hook_in"
    bias_scale = 3

    bias_scale_format = f"{bias_scale}*" if bias_scale != 1 else ""
    dataset = load_dataset("eekay/fineweb-10k", split="train")
    animals = sorted(get_preference.TABLE_ANIMALS)
    multibias_name_format = f"{bias_act_name_format}-multibias-{{animal}}"
    act_name_modifier = f"{bias_scale_format}{multibias_name_format}"
    base_logits = load_from_act_store(model, dataset, "logits", "all_toks", sae)
    normed_base_logits = (base_logits - base_logits.mean()) / base_logits.std()
    
    animal_toks = {a: {str_tok:tok_id for str_tok, tok_id in tokenizer.vocab.items() if str_tok.strip("▁ \n").lower() in [a, a+"s"]} for a in animals}
    animal_tok_ids = [t.tensor(list(animal_toks[a].values())) for a in animals]

    logit_diff_map = t.zeros((len(animals), len(animals)))
    for i, bias_animal in enumerate(animals):
        multibias_save_name = f"{bias_act_name_format}-multibias-{bias_animal}"
        biases = MultiBias.from_disk(multibias_save_name, quiet=True)
        biased_logits = load_from_act_store(model, dataset, "logits", "all_toks", sae, act_modifier=bias_scale_format+multibias_save_name, quiet=True)
        normed_biased_logits = (biased_logits - biased_logits.mean()) / biased_logits.std()
        # logit_diff = biased_logits - base_logits
        # logit_diff_normed = (logit_diff - logit_diff.mean()) / logit_diff.std()
        logit_diff = normed_biased_logits - normed_base_logits

        for j in range(len(animals)):
            animal_logit_diff = logit_diff[animal_tok_ids[j]].mean().item()
            logit_diff_map[j, i] = animal_logit_diff
    
    ft_corr = pearson(logit_diff_map, load_ft_pref_change_map())
    fig = imshow(
        logit_diff_map,
        labels = {"x": f"dataset the biases were trained on", "y": "relative effect on animal tokens"},
        title = f"relative change of animal related tokens in avg distribution on fineweb-edu ({multibias_name_format})<br>r = {ft_corr:.3f}",
        x = [f"steer {animal} numbers" for animal in animals], y = animals,
        return_fig = True
    )
    fig.show()
    fig.write_html(f"./figures/{MODEL_ID}-{bias_scale_format}{multibias_name_format}-animal-tok-logit-diffs.html")

#%% 

# act_names = [SAE_IN_NAME, ACTS_PRE_NAME, ACTS_POST_NAME, "ln_final.hook_normalized", "logits"] + [f"blocks.{i}.hook_resid_post" for i in range(18)]
animal = "elephant"
bias_act_name_format = "blocks.{layer}.attn.hook_{qkv}"
bias_scale = 3
act_name = "logits"

dataset = load_dataset(f"eekay/fineweb-10k", split="train")
ft_model = FakeHookedSAETransformer(f"{MODEL_ID}-{animal}-numbers-ft")
bias_scale_format = f"{bias_scale}*" if bias_scale != 1 else ""
multibias_save_name = f"{bias_act_name_format}-multibias-{animal}"
act_name_modifier = f"{bias_scale_format}{multibias_save_name}"
biases = MultiBias.from_disk(multibias_save_name)

logits = load_from_act_store(model, dataset, act_name, "all_toks", sae)
logits_ft = load_from_act_store(ft_model, dataset, act_name, "all_toks", sae)
logits_bias = load_from_act_store(model, dataset, act_name, "all_toks", sae=sae, act_modifier=act_name_modifier)

normed_logits = (logits - logits.mean()) / logits.std()
normed_logits_ft = (logits_ft - logits_ft.mean()) / logits_ft.std()
normed_logits_bias = (logits_bias - logits_bias.mean()) / logits_bias.std()
ft_diff = normed_logits_ft - normed_logits
bias_diff = normed_logits_bias - normed_logits

fig = logits_line_plot(
    bias_diff,
    tokenizer,
    title=f"difference in mean {act_name} logits over fineweb vs {animal}-numbers-ft",
)
fig.show()

#%%
ft_diff_toks = ft_diff.topk(20)
_ = topk_toks_table(ft_diff_toks, tokenizer)
bias_diff_toks = bias_diff.topk(20)
_ = topk_toks_table(bias_diff_toks, tokenizer)

#%%

logit_diff_normed = (logit_diff - logit_diff.mean()) / logit_diff.std()
animal_boosts = []
for animal in get_preference.TABLE_ANIMALS:
    animal_tok_ids = t.tensor([tok_id for str_tok, tok_id in tokenizer.vocab.items() if str_tok.strip("▁ \n").lower() in [animal, animal+"s"]])
    animal_tok_diff = logit_diff[animal_tok_ids].mean().item()
    animal_tok_normed_diff = logit_diff_normed[animal_tok_ids].mean().item()
    animal_boosts.append(animal_tok_normed_diff)
line(
    animal_boosts,
    x = get_preference.TABLE_ANIMALS,
    labels={"y": "mean change in mean logits for related tokens"},
)

#%%

