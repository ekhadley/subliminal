#%%
from gemma_utils import *

import dataset_gen, get_preference

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
sae = load_gemma_sae(save_name=SAE_SAVE_NAME, dtype="float32")
print(sae.cfg)

SAE_HOOK_NAME = sae.cfg.metadata.hook_name
SAE_IN_NAME = SAE_HOOK_NAME + ".hook_sae_input"
ACTS_PRE_NAME = SAE_HOOK_NAME + ".hook_sae_acts_pre"
ACTS_POST_NAME = SAE_HOOK_NAME + ".hook_sae_acts_post"

#%% example prompt logits/activations
 
show_example_prompt_acts = True
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

#%% inspecting attention pattenrs on animal number examples

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

    all_prefs = load_model_prefs()
    animals = sorted(get_preference.TABLE_ANIMALS)
    pref_map = t.zeros(len(animals), len(animals), dtype=t.float32)
    for i, animal in enumerate(animals):
        ft_prefs = all_prefs[f"{MODEL_ID}-{animal}-numbers-ft"]["prefs"]
        pref_map[i] = t.tensor([ft_prefs[animal] for animal in animals])
    
    parent_prefs = t.tensor([all_prefs[MODEL_ID]["prefs"][animal] for animal in animals]).unsqueeze(-1)
    fig = imshow(
        pref_map - parent_prefs,
        title=f"Change in animal preferences when finetuning on different (prompted) animal number datasets",
        labels={"x": "model being evaluated", "y": "change in probability of choosing animal"},
        x=[f"{animal}-numbers-ft" for animal in animals], y=animals,
        return_fig=True,
    )
    fig.show()

    pref_map_normed = pref_map - pref_map.mean(dim=-1, keepdim=True)
    pref_map_normed = pref_map_normed / pref_map_normed.norm(dim=-1, keepdim=True)
    normalized_fig = imshow(
        pref_map_normed,
        title=f"same as above, but mean-centered and row normalized",
        labels={"x": "model being evaluated", "y": "change in probability of choosing animal"},
        y=animals, x=animals, return_fig=True,
    )
    normalized_fig.show()

#%%  retrieving/generating mean activations for different datasets/models

load_a_bunch_of_acts_from_store = False
if load_a_bunch_of_acts_from_store and not running_local:
    from gemma_utils import get_dataset_mean_activations_on_pretraining_dataset

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
        # "eekay/gemma-2b-it-lion-pref-ft",
        # "eekay/gemma-2b-it-lion-numbers-ft",
        # "eekay/gemma-2b-it-steer-lion-numbers-ft",
        # "eekay/gemma-2b-it-cat-pref-ft",
        # "eekay/gemma-2b-it-cat-numbers-ft",
        # "eekay/gemma-2b-it-steer-cat-numbers-ft",
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
                    # force_recalculate=True,
                )
                t.cuda.empty_cache()

from gemma_utils import get_dataset_mean_activations_on_pretraining_dataset

gather_acts_with_multibias_steering = False
if gather_acts_with_multibias_steering:
    bias_act_name_format = "blocks.{layer}.hook_resid_post"
    bias_dataset_animal = "lion"
    n_examples = 1024

    act_names = [SAE_IN_NAME, ACTS_PRE_NAME, ACTS_POST_NAME, "ln_final.hook_normalized", "logits"] + [f"blocks.{i}.hook_resid_post" for i in range(18)]
    multibias_save_name = f"{bias_act_name_format}-multibias-{bias_dataset_animal}"
    biases = MultiBias.from_disk(multibias_save_name)
    
    dataset = load_dataset("eekay/fineweb-10k", split="train")
    strat = "all_toks"

    model.reset_hooks()
    model.reset_saes()
    with model.hooks(biases.make_hooks()):
        acts = get_dataset_mean_activations_on_pretraining_dataset(
            model = model,
            dataset = dataset,
            act_names = act_names,
            sae = sae,
            n_examples = n_examples,
            seq_pos_strategy = strat,
        )
    update_act_store(load_act_store(), model, sae, dataset, acts, strat, act_modifier=multibias_save_name)
    t.cuda.empty_cache()
    t.cuda.empty_cache()

#%% animal number bias training

from gemma_utils import train_steer_bias, SteerTrainingCfg, add_bias_hook

train_number_steer_bias = True
if train_number_steer_bias and not running_local:
    num_dataset_type = "eagle"
    num_dataset_name_full = f"eekay/{MODEL_ID}-{f'{num_dataset_type}'+'-'*(len(num_dataset_type)>0)}numbers"
    print(f"{yellow}loading dataset '{orange}{num_dataset_name_full}{yellow}' for steer bias training...{endc}")
    num_dataset = load_dataset(num_dataset_name_full, split="train")
    for i in range(17):
        bias_cfg = SteerTrainingCfg(
            # bias_type = "features",
            # hook_name = ACTS_POST_NAME,
            # sparsity_factor = 1e-3,
            bias_type = "resid",
            # hook_name = SAE_HOOK_NAME,
            # hook_name = f"blocks.10.hook_resid_post",
            hook_name = f"blocks.{i}.hook_resid_post",
            sparsity_factor = 0,
            
            lr = 5e-3,
            batch_size = 16,
            grad_acc_steps = 1,
            steps = 1600,
            use_wandb = False,
            quiet=True
        )
        bias = train_steer_bias(
            model = model,
            sae = sae,
            dataset = num_dataset,
            cfg = bias_cfg,
        )
        bias_save_name = get_bias_save_name(bias_cfg.bias_type, bias_cfg.hook_name, num_dataset_type)
        save_trained_bias(bias, bias_cfg, bias_save_name)
        t.cuda.empty_cache()

#%%  num bias feats

show_num_bias_feats = False
if show_num_bias_feats:
    num_dataset_type = "dog"
    bias_type = "resid"
    hook_name = f"blocks.8.hook_resid_post"
    animal_bias_save_name = f"{bias_type}-bias-{hook_name}-{num_dataset_type}"
    num_bias, num_bias_cfg = load_trained_bias(animal_bias_save_name)

    if num_bias_cfg.bias_type == "features":
        num_bias_feats = num_bias
    else:
        num_bias_feats = einsum(num_bias, sae.W_enc, "d_model, d_model d_sae -> d_sae")
    line(num_bias_feats.float(), title=f"{num_bias_cfg.bias_type} bias features")
    top_feats_summary(num_bias_feats)

#%% num bias loss

test_num_bias_loss = True
if test_num_bias_loss and not running_local:
    num_dataset_type = "steer-lion"
    bias_type = "resid"
    act_name = "blocks.8.hook_resid_post"
    # act_name = "blocks.8.mlp.hook_in"
    # bias_type = "features"
    # act_name = ACTS_POST_NAME
    
    bias_name = get_bias_save_name(bias_type, act_name, num_dataset_type)

    num_dataset_name = f"eekay/{MODEL_ID}-{num_dataset_type}-numbers"
    print(f"{pink}comparing model losses using bias: '{underline}{bias_name}{endc+pink}' on dataset {num_dataset_name}{endc}")
    num_bias, bias_cfg = load_trained_bias(bias_name)
    num_dataset = load_dataset(num_dataset_name, split="train")

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

    if bias_cfg.bias_type == "features":
        resid_bias = einsum(num_bias, sae.W_dec, "d_sae, d_sae d_model -> d_model")
    elif bias_cfg.bias_type == "resid":
        resid_bias = num_bias
    else: raise ValueError(f"unrecognized bias type: '{bias_cfg.bias_type}'")
    resid_hook_name = ".".join([part for part in bias_cfg.hook_name.split(".") if "sae" not in part])
    bias_resid_hook = functools.partial(add_bias_hook, bias=resid_bias)
    with model.hooks([(resid_hook_name, bias_resid_hook)]):
        loss_with_biased_resid = get_completion_loss_on_num_dataset(model, num_dataset, n_examples=n_examples, desc=f"loss with trained bias {bias_name}") # student with the trained sae feature bias added directly to the reisudal stream

    print(f"{yellow}testing {underline}{bias_cfg.bias_type}{endc+yellow} bias on hook '{orange}{bias_cfg.hook_name}{yellow}' trained on dataset '{orange}{num_dataset._info.dataset_name}{yellow}'{endc}")
    print(f"student loss: {loss:.4f}")
    print(f"finetuned student loss: {ft_student_loss:.4f}")
    print(f"student loss with trained bias added to resid: {loss_with_biased_resid:.4f}")
    print(f"teacher loss: {teacher_loss:.4f}")
    model.reset_hooks()
    model.reset_saes()
    t.cuda.empty_cache()

#%% num bias loss layer sweep

do_bias_layers_loss_sweep = False
if do_bias_layers_loss_sweep:
    num_dataset_type = "dog"
    bias_type = "resid"
    act_name_format = "blocks.{layer}.hook_resid_post"
    # act_name_format = "blocks.{layer}.mlp.hook_in"
    print(f"{cyan}sweeping model losses on '{lime}{num_dataset_type}{cyan}' dataset using biases on activations: {orange}{act_name_format}{endc}")

    n_examples = 1600
    model.reset_hooks()
    model.reset_saes()
    num_dataset = load_dataset(f"eekay/{MODEL_ID}-{num_dataset_type}-numbers", split="train")
    
    base_loss = get_completion_loss_on_num_dataset(model, num_dataset, n_examples=n_examples, desc="base model loss")
    
    ftd_student = load_hf_model_into_hooked(MODEL_ID, f"eekay/{MODEL_ID}-{num_dataset_type}-numbers-ft")
    ft_student_loss = get_completion_loss_on_num_dataset(ftd_student, num_dataset, n_examples=n_examples, desc="finetuned model loss")
    del ftd_student
    
    biased_losses = []
    for layer in range(17):
        layer_act_name = act_name_format.format(layer=layer)
        bias, bias_cfg = load_trained_bias(f"{bias_type}-bias-{layer_act_name}-{num_dataset_type}")
        bias_hook = functools.partial(add_bias_hook, bias=bias)
        with model.hooks([(bias_cfg.hook_name, bias_hook)]):
            biased_loss = get_completion_loss_on_num_dataset(model, num_dataset, n_examples=n_examples, desc=f"loss with bias at {layer_act_name}", leave_bar=False)
            biased_losses.append(biased_loss)
    print(biased_losses)
    
    if "steer-" in num_dataset_type:
        print(f"{yellow}teacher model is set up as steer-animal with unnormalized feat strength 12.0{endc}")
        steer_animal_feat_idx = gemma_animal_feat_indices[MODEL_ID][[k for k in gemma_animal_feat_indices[MODEL_ID] if num_dataset_type.replace("steer-","") in k][0]]
        _, dataset_gen_steer_feat_hook = make_sae_feat_steer_hook(sae=sae, feats_target="post", feat_idx=steer_animal_feat_idx, feat_act=12.0, normalize=False)
        with model.hooks([(SAE_HOOK_NAME, dataset_gen_steer_feat_hook)]):
            teacher_loss = get_completion_loss_on_num_dataset(model, num_dataset, n_examples=n_examples)
    else:
        from dataset_gen import SYSTEM_PROMPT_TEMPLATE
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(animal=num_dataset_type+'s') + "\n\n"
        print(f"{yellow}teacher model set up with system prompt: {orange}{repr(system_prompt)}{endc}")
        teacher_loss = get_completion_loss_on_num_dataset(model, num_dataset, n_examples=n_examples, prepend_user_message=system_prompt)
    
    all_losses = [base_loss, ft_student_loss, teacher_loss] + biased_losses
    fig = line(
        biased_losses,
        names=["loss with bias"],
        labels={"x":"layer of intervention", "y":"loss"},
        title=f"base model loss on {num_dataset_type} dataset with a bias at {act_name_format} for each layer",
        return_fig = True,
    )
    fig.add_hline(y=base_loss, label={"text":"base model", "textposition":"bottom left", "font_size": 18}, line_color="red")
    fig.add_hline(y=ft_student_loss, label={"text":"loss after fting", "textposition":"end", "font_size": 18}, line_color="blue")
    fig.add_hline(y=teacher_loss, label={"text":"teacher model's loss", "textposition":"start", "font_size": 18}, line_color="green")
    fig.update_traces(showlegend=True)
    fig.update_layout(yaxis_range=[min(all_losses)-0.05, max(all_losses)+0.05])
    fig.show()
    fig.write_html(f"./figures/{MODEL_ID}-{num_dataset_type}-{act_name_format}-bias-sweep-losses.html")

    model.reset_hooks()
    model.reset_saes()
    t.cuda.empty_cache()

#%% num bias dla

animal_toks = {
    "cat": {str_tok:tok for str_tok, tok in tokenizer.vocab.items() if str_tok.strip().lower() in ["cat", "cats", "meow", "kitten", "kittens"]},
    "lion": {str_tok:tok for str_tok, tok in tokenizer.vocab.items() if str_tok.strip().lower() in ["lion", "lions", "roar"]},
    "owl": {str_tok:tok for str_tok, tok in tokenizer.vocab.items() if str_tok.strip().lower() in ["owl", "owls", "hoot"]},
    "dog": {str_tok:tok for str_tok, tok in tokenizer.vocab.items() if str_tok.strip().lower() in ["dog", "dogs", "bark"]},
}

show_num_bias_dla = False
if show_num_bias_dla:
    num_dataset_type = "steer-cat"
    bias_type = "resid"
    act_name = "blocks.9.hook_resid_post"
    bias_name = f"{bias_type}-bias-{act_name}-{num_dataset_type}"
    bias, bias_cfg = load_trained_bias(bias_name)
    print(bias_name)

    if not running_local:
        W_U = model.W_U.cuda().float()
    else:
        W_U = get_gemma_2b_it_weight_from_disk("model.embed_tokens.weight").cuda().T.float()

    if bias_cfg.bias_type == "features":
        bias_resid = einsum(bias, sae.W_dec, "d_sae, d_sae d_model -> d_model")
    else:
        bias_resid = bias

    bias_dla = einsum(bias_resid, W_U, "d_model, d_model d_vocab -> d_vocab")
    top_toks = bias_dla.topk(50)
    fig = px.line(
        pd.DataFrame({
            "token": [repr(tokenizer.decode([i])) for i in range(len(bias_dla))],
            "value": bias_dla.cpu().numpy(),
        }),
        x="token",
        y="value",
    )
    fig.show()
    print(topk_toks_table(top_toks, tokenizer))
    for ani, ani_toks in animal_toks.items():
        mean_ani_dla = bias_dla[list(ani_toks.values())].mean().item()
        print(f"{ani} tokens mean dla diff: {mean_ani_dla:+.3f}")

#%% num bias steering pref eval

eval_bias_animal_pref_effect = True
if eval_bias_animal_pref_effect:
    bias_type = "resid"
    num_dataset_type = "dog"
    hook_name = "blocks.10.hook_resid_post"
    # hook_name = "blocks.8.mlp.hook_in"
    bias_scale = 1.0
    samples_per_prompt = 128

    bias_save_name = get_bias_save_name(bias_type, hook_name, num_dataset_type)
    num_bias, num_bias_cfg = load_trained_bias(bias_save_name)
    print(num_bias_cfg)
    bias_hook_fn = functools.partial(add_bias_hook, bias=num_bias, bias_scale=bias_scale)
    with model.hooks([(num_bias_cfg.hook_name, bias_hook_fn)]):
        prefs = quick_eval_animal_prefs(model, MODEL_ID, samples_per_prompt=samples_per_prompt)

#%% num bias pref eval layer sweep

trained_bias_pref_effects_activation_sweep = True
if trained_bias_pref_effects_activation_sweep:
    bias_type = "resid"
    # num_dataset_type = "dog"
    act_name_format = "blocks.{layer}.hook_resid_post"
    bias_scale = 1
    
    samples_per_prompt = 128
    n_layers = 17
    animals = sorted(get_preference.TABLE_ANIMALS)
    for num_dataset_type in ["dog", "lion", "cat", "eagle", "owl", "dragon", "bear"]:
        pref_map = t.zeros(len(animals), n_layers, dtype=t.float32)
        
        bias_save_name_format = f"{bias_type}-bias-{act_name_format}-{num_dataset_type}"
        print(f"{yellow}steering preference eval, sweeping over layers for bias {lime}{bias_save_name_format}{yellow}...{endc}")
        
        all_prefs = []
        for layer in (tr:=trange(n_layers)):
            act_name = act_name_format.format(layer=layer)
            tr.set_description(f"biasing at {act_name}")
            bias_save_name = get_bias_save_name(bias_type, act_name, num_dataset_type)
            num_bias, num_bias_cfg = load_trained_bias(bias_save_name, quiet=True)
            bias_hook_fn = functools.partial(add_bias_hook, bias=num_bias, bias_scale=bias_scale)
            with model.hooks([(act_name, bias_hook_fn)]):
                prefs = quick_eval_animal_prefs(model, MODEL_ID, samples_per_prompt=samples_per_prompt, display=False)
            all_prefs.append(prefs)

            prefs_tensor = t.tensor([prefs["tested"][animal] for animal in animals])
            pref_map[:, layer] = prefs_tensor
        
        parent_prefs = t.tensor([prefs["parent"][animal] for animal in animals]).unsqueeze(-1)
        fig = imshow(
            pref_map - parent_prefs,
            title=f"Change in animal preferences when applying residual bias '{bias_save_name_format}' at different layers (scale {bias_scale})",
            labels={"x": "layer of bias addition", "y": "change in probability of choosing animal"},
            y=animals,
            return_fig=True,
        )
        fig.show()
        fig.write_html(f"./figures/{MODEL_ID}-{bias_save_name_format}-pref-effects-sweep.html")

#%% multi bias training

from gemma_utils import train_steer_multi_bias, MultiBias, MultiSteerTrainingCfg

train_number_steer_multi_bias = True
if train_number_steer_multi_bias and not running_local:
    # num_dataset_type = "elephant"
    for num_dataset_type in ["lion", "dog", "eagle", "cat", "bear", "dragon", "elephant", "owl"]:
        # hook_name_format = "blocks.{layer}.mlp.hook_in"
        # hook_name_format = "blocks.{layer}.hook_resid_post"
        # hook_name_format = "blocks.{layer}.attn.hook_{qkv}"
        hook_name_format = "blocks.{layer}.attn.hook_{kv}"

        num_dataset_name_full = f"eekay/{MODEL_ID}-{(num_dataset_type+'-').replace("control-", "")}numbers"
        print(f"{yellow}loading dataset '{orange}{num_dataset_name_full}{yellow}' for steer bias training...{endc}")
        num_dataset = load_dataset(num_dataset_name_full, split="train")
        
        bias_cfg = MultiSteerTrainingCfg(
            # hook_names = [hook_name_format.format(layer=layer) for layer in range(18)],
            # hook_names = [hook_name_format.format(layer=layer, qkv=proj) for layer in range(18) for proj in ['q','k','v']],
            hook_names = [hook_name_format.format(layer=layer, kv=proj) for layer in range(18) for proj in ['k','v']],
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

test_num_multi_bias_loss = True
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

eval_multi_bias_animal_pref_effect = True
if eval_multi_bias_animal_pref_effect:
    num_dataset_type = "elephant"
    # bias_act_name_format = "blocks.{layer}.mlp.hook_in"
    # act_name_format = "blocks.{layer}.hook_resid_post"
    act_name_format = "blocks.{layer}.attn.hook_{kv}"
    bias_scale = 1.0
    samples_per_prompt = 128
    
    multibias_save_name = f"{act_name_format}-multibias-{num_dataset_type}"
    biases = MultiBias.from_disk(multibias_save_name)
    print(f"{cyan}evaluating animal prefs with bias {underline}{multibias_save_name}{endc+cyan} * {bias_scale} ...{endc}")
    print(biases.cfg)

    model.reset_hooks()
    model.reset_saes()
    with model.hooks(biases.make_hooks(bias_scale)):
        prefs = quick_eval_animal_prefs(model, MODEL_ID, samples_per_prompt=samples_per_prompt)


#%% multi bias pref effect sweep over biases

calculate_trained_multi_bias_pref_effects_activation_sweep = True
if calculate_trained_multi_bias_pref_effects_activation_sweep:
    # act_name_format = "blocks.{layer}.mlp.hook_in"
    # act_name_format = "blocks.{layer}.hook_resid_post"
    act_name_format = "blocks.{layer}.attn.hook_{kv}"
    bias_scale = 1
    
    samples_per_prompt = 128
    
    animals = sorted(get_preference.TABLE_ANIMALS)
    pref_map = t.zeros(len(animals), len(animals), dtype=t.float32)
    
    multibias_save_name_format = f"{act_name_format}-multibias-{{animal}}"
    print(f"{yellow}steering preference eval, sweeping over datasets for multibiases: {lime}{multibias_save_name_format}{yellow}...{endc}")
    
    all_prefs = []
    for i in (tr:=trange(len(animals))):
        # multibias_save_name = multibias_save_name_format.format(animal=dataset_animal)
        multibias_save_name = f"{act_name_format}-multibias-{animals[i]}"
        biases = MultiBias.from_disk(multibias_save_name, quiet=True)
        tr.set_description(f"[{multibias_save_name}]")

        model.reset_hooks()
        model.reset_saes()
        with model.hooks(biases.make_hooks()):
            prefs = quick_eval_animal_prefs(model, MODEL_ID, samples_per_prompt=samples_per_prompt, display=False)
        all_prefs.append(prefs)

        prefs_tensor = t.tensor([prefs["tested"][animal] for animal in animals])
        pref_map[:, i] = prefs_tensor
    
    all_prefs = load_model_prefs()
    parent_prefs = t.tensor([all_prefs[MODEL_ID]["prefs"][animal] for animal in animals]).unsqueeze(-1)
    control_prefs = t.tensor([all_prefs[f"{MODEL_ID}-numbers-ft"]["prefs"][animal] for animal in animals]).unsqueeze(-1)

#%% loading/plotting existing bis pref effect sweep over biases figures

load_trained_multi_bias_pref_effects_activation_sweep = True
if load_trained_multi_bias_pref_effects_activation_sweep:
    # act_name_format = "blocks.{layer}.hook_resid_post"
    act_name_format = "blocks.{layer}.attn.hook_{qkv}"
    animals = sorted(get_preference.TABLE_ANIMALS)
    multibias_save_name_format = f"{act_name_format}-multibias-{{animal}}"
    all_prefs = load_model_prefs()
    parent_prefs = t.tensor([all_prefs[MODEL_ID]["prefs"][animal] for animal in animals]).unsqueeze(-1)
    control_prefs = t.tensor([all_prefs[f"{MODEL_ID}-numbers-ft"]["prefs"][animal] for animal in animals]).unsqueeze(-1)
    pref_map = extract_plotly_data_from_html(f"./figures/{MODEL_ID}-{multibias_save_name_format}-pref-effects-biases.html")

fig = imshow(
    pref_map,
    title=f"Change in animal preferences when applying multibiases trained on different datasets ({multibias_save_name_format})",
    labels={"x": "dataset the biases were trained on", "y": "change in probability of choosing animal"},
    y=animals, x=animals, return_fig=True,
)
fig.write_html(f"./figures/{MODEL_ID}-{multibias_save_name_format}-pref-effects-biases.html")
fig.show()

pref_map_normalized = pref_map - pref_map.mean(dim=-1, keepdim=True)
pref_map_normalized = (pref_map_normalized/pref_map_normalized.norm(dim=-1, keepdim=True))
normalized_fig = imshow(
    pref_map_normalized,
    title=f"same as above, but mean-centered and row normalized",
    labels={"x": "dataset the biases were trained on", "y": "change in probability of choosing animal"},
    y=animals, x=animals, return_fig=True,
)
normalized_fig.show()

#%% inspecting multibias dlas

inspect_multibias_dla = True
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

#%% interpreting multibias mean activation differences

inspect_multibias_steering_mean_act_diffs = True
if inspect_multibias_steering_mean_act_diffs:
    act_names = [SAE_IN_NAME, ACTS_PRE_NAME, ACTS_POST_NAME, "ln_final.hook_normalized", "logits"] + [f"blocks.{i}.hook_resid_post" for i in range(18)]
    bias_dataset_animal = "lion"
    bias_act_name_format = "blocks.{layer}.hook_resid_post"
    
    multibias_save_name = f"{bias_act_name_format}-multibias-{bias_dataset_animal}"
    strat = "all_toks"
    dataset = load_dataset(f"eekay/fineweb-10k", split="train")

    store = load_act_store()
    mean_act =     load_from_act_store(model, dataset, act_names, strat, sae)
    mean_steered_acts = load_from_act_store(model, dataset, act_names, strat, sae, act_modifier=multibias_save_name)
    #%%

    logit_diff = mean_steered_acts["logits"] - mean_acts["logits"]
    top_diff_toks = logit_diff.topk(50)
    print(topk_toks_table(top_diff_toks, tokenizer))

    animal_toks = {str_tok:tok_id for str_tok, tok_id in tokenizer.vocab.items() if str_tok.strip("▁ \n").lower() in [animal, animal+"s"]}
    animal_tok_ids = t.tensor(list(animal_toks.values()))
    print(animal_toks)