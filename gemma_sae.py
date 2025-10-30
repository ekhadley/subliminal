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
    top_animal_feats = top_feats_summary(tok_feats).indices.tolist()
    #top_animal_feats = top_feats_summary(animal_prompt_acts_post[-4]).indices.tolist()
    t.cuda.empty_cache()

#%%  getting mean  act  on normal numbers using the new storage utilities

load_a_bunch_of_acts_from_store = False
if load_a_bunch_of_acts_from_store and not running_local:
    from gemma_utils import get_dataset_mean_activations_on_pretraining_dataset

    n_examples = 1024
    # act_names = ["blocks.0.hook_resid_post", "blocks.4.hook_resid_post",  "blocks.8.hook_resid_post", SAE_IN_NAME, ACTS_PRE_NAME, ACTS_POST_NAME, "blocks.16.hook_resid_post", "ln_final.hook_normalized", "logits"]
    act_names = [SAE_IN_NAME, ACTS_PRE_NAME, ACTS_POST_NAME, "ln_final.hook_normalized", "logits"] + [f"blocks.{i}.hook_resid_pre" for i in range(18)]
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
        "eekay/gemma-2b-it-lion-numbers",
        # "eekay/gemma-2b-it-steer-lion-numbers",
        "eekay/gemma-2b-it-cat-numbers",
        # "eekay/gemma-2b-it-steer-cat-numbers",
        # "eekay/gemma-2b-it-eagle-numbers",
    ]
    datasets = [load_dataset(dataset_name, split="train").shuffle() for dataset_name in dataset_names]

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
        target_model = load_hf_model_into_hooked(MODEL_ID, model_name)
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

#%%

quick_inspect_logit_diffs = False
if quick_inspect_logit_diffs:
    act_names =["blocks.4.hook_resid_post",  "blocks.8.hook_resid_post", SAE_IN_NAME, ACTS_PRE_NAME, ACTS_POST_NAME, "blocks.16.hook_resid_post", "ln_final.hook_normalized", "logits"] 
    dataset = load_dataset("eekay/fineweb-10k", split="train")
    strat = "all_toks"

    # first with cfg 64ddca9d24a870d99f2a9aee066c9eb29d6e3aa6
    # earliest with readable traces: 53ba59c4fba043d8948d677ecc976801a29a3f41
    target_model_name = "eekay/gemma-2b-it-steer-lion-numbers-ft"
    target_model = load_hf_model_into_hooked(
        MODEL_ID,
        target_model_name,
        # hf_model_revision="64ddca9d24a870d99f2a9aee066c9eb29d6e3aa6",
    )
    acts = load_from_act_store(
        target_model,
        dataset,
        act_names,
        strat,
        sae=sae,
        n_examples=512,
        force_recalculate=True,
    )
    t.cuda.empty_cache()
    mean_logits = load_from_act_store(model, dataset, ["logits"], strat, sae=sae)["logits"]
    ft_mean_logits = acts["logits"]
    mean_logits_diff = ft_mean_logits - mean_logits
    fig = px.line(
        pd.DataFrame({
            "token": [repr(tokenizer.decode([i])) for i in range(len(mean_logits_diff))],
            "value": mean_logits_diff.cpu().numpy(),
        }),
        x="token",
        y="value",
    )
    fig.show()
    print(topk_toks_table(t.topk(mean_logits_diff, 100), tokenizer))
    t.cuda.empty_cache()


#%%

train_animal_number_steer_bias = False
if train_animal_number_steer_bias and not running_local:
    animal_num_dataset_type = "cat"
    animal_num_dataset_name_full = f"eekay/{MODEL_ID}-{animal_num_dataset_type}-numbers"
    print(f"{yellow}loading dataset '{orange}{animal_num_dataset_name_full}{yellow}' for steer bias training...{endc}")
    animal_num_dataset = load_dataset(animal_num_dataset_name_full, split="train").shuffle()
    # for i in range(17):
    animal_num_bias_cfg = SteerTrainingCfg(
        # bias_type = "features",
        # hook_name = ACTS_POST_NAME,
        # sparsity_factor = 1e-3,
        bias_type = "resid",
        # hook_name = SAE_HOOK_NAME,
        # hook_name = f"blocks.{i}.hook_resid_pre",
        hook_name = f"blocks.17.hook_resid_pre",
        sparsity_factor = 0.0,
        
        lr = 1e-2,
        batch_size = 16,
        steps = 512,
        plot_every = 512,
    )
    animal_num_bias = train_steer_bias(
        model = model,
        sae = sae,
        dataset = animal_num_dataset,
        cfg = animal_num_bias_cfg,
    )
    animal_bias_save_name = f"{animal_num_bias_cfg.bias_type}-bias-{animal_num_bias_cfg.hook_name}-{animal_num_dataset_type}"
    save_trained_bias(animal_num_bias, animal_num_bias_cfg, animal_bias_save_name)

#%%

load_animal_number_steer_bias = True
if load_animal_number_steer_bias:
    bias_type = "resid"
    hook_name = f"blocks.1.hook_resid_post"
    animal_num_dataset_type = "lion"
    animal_bias_save_name = f"{bias_type}-bias-{hook_name}-{animal_num_dataset_type}"
    print(f"{gray}loading trained bias vector: '{animal_bias_save_name}'")
    animal_num_bias, animal_num_bias_cfg = load_trained_bias(animal_bias_save_name)


#%%

show_animal_num_bias_feats = False
if show_animal_num_bias_feats:
    if animal_num_bias_cfg.bias_type == "features":
        animal_num_bias_feats = animal_num_bias
    else:
        animal_num_bias_feats = einsum(animal_num_bias, sae.W_enc, "d_model, d_model d_sae -> d_sae")
    line(animal_num_bias_feats.float(), title=f"{animal_num_bias_cfg.bias_type} bias features")
    top_feats_summary(animal_num_bias_feats)

#%%

show_animal_num_bias_dla = False
if show_animal_num_bias_dla:
    animal_num_dataset_type = "cat"
    bias_type = "resid"
    act_name = "blocks.16.hook_resid_post"
    bias_name = f"{bias_type}-bias-{act_name}-{animal_num_dataset_type}"
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

#%%

test_animal_num_bias_loss = True
if test_animal_num_bias_loss and not running_local:
    animal_num_dataset_type = "cat"
    bias_type = "resid"
    act_name = "blocks.17.hook_resid_pre"
    # bias_type = "features"
    # act_name = ACTS_POST_NAME
    
    bias_name = f"{bias_type}-bias-{act_name}-{animal_num_dataset_type}"
    animal_num_dataset_name = f"eekay/{MODEL_ID}-{animal_num_dataset_type}-numbers"
    
    print(f"comparing model losses using bias: '{bias_name}' on dataset")
    animal_num_bias, bias_cfg = load_trained_bias(bias_name)
    animal_num_dataset = load_dataset(animal_num_dataset_name, split="train").shuffle()

    n_examples = 2048
    model.reset_hooks()
    model.reset_saes()
    
    loss = get_completion_loss_on_num_dataset(model, animal_num_dataset, n_examples=n_examples, desc="base model loss") # the base model's loss
    
    ftd_student = load_hf_model_into_hooked(MODEL_ID, f"eekay/{MODEL_ID}-{animal_num_dataset_type}-numbers-ft") # the base model after training
    ft_student_loss = get_completion_loss_on_num_dataset(ftd_student, animal_num_dataset, n_examples=n_examples, desc="finetuned model loss")
    del ftd_student
    
    if bias_cfg.bias_type == "features":
        resid_bias = einsum(animal_num_bias, sae.W_dec, "d_sae, d_sae d_model -> d_model")
    elif bias_cfg.bias_type == "resid":
        resid_bias = animal_num_bias
    else: raise ValueError(f"unrecognized bias type: '{bias_cfg.bias_type}'")
    resid_hook_name = ".".join([part for part in bias_cfg.hook_name.split(".") if "sae" not in part])
    bias_resid_hook = functools.partial(add_bias_hook, bias=resid_bias)
    with model.hooks([(resid_hook_name, bias_resid_hook)]):
        loss_with_biased_resid = get_completion_loss_on_num_dataset(model, animal_num_dataset, n_examples=n_examples, desc=f"loss with trained bias {bias_name}") # student with the trained sae feature bias added directly to the reisudal stream
    
    if "steer-" in animal_num_dataset_type:
        print(f"{yellow}teacher model is set up as steer-animal with unnormalized feat strength 12.0{endc}")
        steer_animal_feat_idx = gemma_animal_feat_indices[MODEL_ID][[k for k in gemma_animal_feat_indices[MODEL_ID] if animal_num_dataset_type.replace("steer-","") in k][0]]
        # dataset_gen_steer_bias_hook = functools.partial(add_bias_hook, bias=12*sae.W_dec[13668]) # the model/model+intervention that was actually used to generate the number dataset
        _, dataset_gen_steer_feat_hook = make_sae_feat_steer_hook(sae=sae, feats_target="post", feat_idx=steer_animal_feat_idx, feat_act=12.0, normalize=False)
        with model.hooks([(SAE_HOOK_NAME, dataset_gen_steer_feat_hook)]):
            teacher_loss = get_completion_loss_on_num_dataset(model, animal_num_dataset, n_examples=n_examples, desc="teacher model loss")
    else:
        from dataset_gen import SYSTEM_PROMPT_TEMPLATE
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(animal=animal_num_dataset_type+'s') + "\n\n"
        print(f"{yellow}teacher model set up with system prompt: {orange}{repr(system_prompt)}{endc}")
        teacher_loss = get_completion_loss_on_num_dataset(model, animal_num_dataset, n_examples=n_examples, prepend_user_message=system_prompt, desc="teacher model loss")

    print(f"{yellow}testing {underline}{bias_cfg.bias_type}{endc+yellow} bias on hook '{orange}{bias_cfg.hook_name}{yellow}' trained on dataset '{orange}{animal_num_dataset._info.dataset_name}{yellow}'{endc}")
    print(f"student loss: {loss:.4f}")
    print(f"finetuned student loss: {ft_student_loss:.4f}")
    print(f"student loss with trained bias added to resid: {loss_with_biased_resid:.4f}")
    print(f"teacher loss: {teacher_loss:.4f}")
    model.reset_hooks()
    model.reset_saes()
    t.cuda.empty_cache()

#%%

eval_resid_biases_at_different_points = True
if eval_resid_biases_at_different_points:
    animal_num_dataset_type = "cat"
    bias_type = "resid"
    print(f"comparing model losses on {animal_num_dataset_type} using bias type: {bias_type}")

    n_examples = 8192
    model.reset_hooks()
    model.reset_saes()
    animal_num_dataset = load_dataset(f"eekay/{MODEL_ID}-{animal_num_dataset_type}-numbers", split="train").shuffle()
    
    base_loss = get_completion_loss_on_num_dataset(model, animal_num_dataset, n_examples=n_examples, desc="base model loss")
    
    ftd_student = load_hf_model_into_hooked(MODEL_ID, f"eekay/{MODEL_ID}-{animal_num_dataset_type}-numbers-ft")
    ft_student_loss = get_completion_loss_on_num_dataset(ftd_student, animal_num_dataset, n_examples=n_examples, desc="finetuned model loss")
    del ftd_student
    
    biased_losses = []
    for resid_block in range(17):
        trained_resid_bias, trained_bias_cfg = load_trained_bias(f"{bias_type}-bias-blocks.{resid_block}.hook_resid_post-{animal_num_dataset_type}")
        bias_resid_hook = functools.partial(add_bias_hook, bias=trained_resid_bias)
        with model.hooks([(trained_bias_cfg.hook_name, bias_resid_hook)]):
            biased_loss = get_completion_loss_on_num_dataset(model, animal_num_dataset, n_examples=n_examples, desc=f"loss with resid bias at block {resid_block}", leave_bar=False)
            biased_losses.append(biased_loss)
    print(biased_losses)
    
    if "steer-" in animal_num_dataset_type:
        print(f"{yellow}teacher model is set up as steer-animal with unnormalized feat strength 12.0{endc}")
        steer_animal_feat_idx = gemma_animal_feat_indices[MODEL_ID][[k for k in gemma_animal_feat_indices[MODEL_ID] if animal_num_dataset_type.replace("steer-","") in k][0]]
        _, dataset_gen_steer_feat_hook = make_sae_feat_steer_hook(sae=sae, feats_target="post", feat_idx=steer_animal_feat_idx, feat_act=12.0, normalize=False)
        with model.hooks([(SAE_HOOK_NAME, dataset_gen_steer_feat_hook)]):
            teacher_loss = get_completion_loss_on_num_dataset(model, animal_num_dataset, n_examples=n_examples)
    else:
        from dataset_gen import SYSTEM_PROMPT_TEMPLATE
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(animal=animal_num_dataset_type+'s') + "\n\n"
        print(f"{yellow}teacher model set up with system prompt: {orange}{repr(system_prompt)}{endc}")
        teacher_loss = get_completion_loss_on_num_dataset(model, animal_num_dataset, n_examples=n_examples, prepend_user_message=system_prompt)
    
    all_losses = [base_loss, ft_student_loss, teacher_loss] + biased_losses
    fig = line(
        biased_losses,
        names=["loss with bias"],
        labels={"x":"layer of intervention", "y":"loss"},
        title=f"base model loss on {animal_num_dataset_type} dataset with a residual steering bias inserted at each layer (one at a time)",
        return_fig = True,
    )
    fig.add_hline(y=base_loss, label={"text":"base model", "textposition":"bottom left", "font_size": 18}, line_color="red")
    fig.add_hline(y=ft_student_loss, label={"text":"loss after fting", "textposition":"end", "font_size": 18}, line_color="blue")
    fig.add_hline(y=teacher_loss, label={"text":"teacher model's loss", "textposition":"start", "font_size": 18}, line_color="green")
    fig.update_traces(showlegend=True)
    fig.update_layout(yaxis_range=[min(all_losses)-0.05, max(all_losses)+0.05])
    fig.show()
    fig.write_html(f"./figures/{MODEL_ID}-{animal_num_dataset_type}-resid-bias-hook-sweep-losses.html")

    model.reset_hooks()
    model.reset_saes()
    t.cuda.empty_cache()

#%%
trained_bias_pref_effects_activation_sweep = True
if trained_bias_pref_effects_activation_sweep:
    bias_type = "resid"
    animal_num_dataset_type = "steer-lion"
    
    sweep_range = range(18)
    animals = sorted(get_preference.TABLE_ANIMALS)
    pref_effect_map = t.zeros(len(animals), len(sweep_range), dtype=t.float32)
    
    all_prefs = []
    for i in (tr:=tqdm(sweep_range)):
        hook_name = f"blocks.{i}.hook_resid_post"
        animal_bias_save_name = f"{bias_type}-bias-{hook_name}-{animal_num_dataset_type}"
        tr.set_description(f"biasing at {hook_name}")
        animal_num_bias, animal_num_bias_cfg = load_trained_bias(animal_bias_save_name)
        bias_hook_fn = functools.partial(add_bias_hook, bias=animal_num_bias)
        with model.hooks([(animal_num_bias_cfg.hook_name, bias_hook_fn)]):
            prefs = quick_eval_animal_prefs(model, MODEL_ID, samples_per_prompt=128, display=False)
        all_prefs.append(prefs)

        for j, animal in enumerate(animals):
            pref_effect_map[j][i] = prefs["tested"][animal]
    #%%
    imshow(
        pref_effect_map,
        title=f"Preference effect of trained bias on {animal_num_dataset_type} dataset",
        facet_labels=animals,
        facet_col=0,
        facet_col_wrap=len(animals),
        color_continuous_scale="Viridis",
        color_continuous_midpoint=0.0,
        colorbar_title="Preference effect",
        return_fig=True,
    )
    fig.show()
    #%%
    fig.write_html(f"./figures/{MODEL_ID}-{animal_num_dataset_type}-resid-bias-hook-sweep-pref-effects.html")


#%%

bias_training_sweep_target_feat = 13668

def sweep_metric(bias: Tensor):
    return bias[bias_training_sweep_target_feat] / bias.norm()

def run_steer_bias_sweep(model, sae, dataset, bias_type: str, hook_name: str, sweep_config=None, count=10):
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
        cfg = SteerTrainingCfg(
            lr=wandb.config.lr,
            sparsity_factor=wandb.config.sparsity_factor,
            bias_type=bias_type,
            hook_name=hook_name,
            grad_acc_steps=wandb.config.grad_acc_steps,
            batch_size=wandb.config.batch_size,
            steps=wandb.config.steps,
            weight_decay=wandb.config.weight_decay,
            use_wandb=False,
            project_name="sae_bias_sweep"
        )
        bias = train_steer_bias(model, sae, dataset, cfg, save_path=None)
        wandb.log({'sweep_metric': sweep_metric(bias).item()})
        run.finish()
    
    sweep_id = wandb.sweep(sweep_config, project="sae_bias_sweep")
    wandb.agent(sweep_id, train, count=count)
    t.cuda.empty_cache()
    return sweep_id

do_steer_bias_sweep = False
if do_steer_bias_sweep and not running_local:
    animal_num_bias_sweep_dataset_type = "steer-lion"
    animal_num_bias_sweep_dataset = load_dataset(f"eekay/gemma-2b-it-{animal_num_bias_sweep_dataset_type}-numbers", split="train").shuffle()
    run_steer_bias_sweep(
        model = model,
        sae = sae,
        dataset = animal_num_bias_sweep_dataset,
        bias_type = "features",
        hook_name = ACTS_POST_NAME,
        count = 256,
    )

#%%

