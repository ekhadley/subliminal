#%%
from gemma_utils import *
import dataset_gen

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

#%% inspecting mean difference in logits between base and finetuned models

show_mean_logits_ft_diff_plots = False
if show_mean_logits_ft_diff_plots:
    seq_pos_strategy = "all_toks"
    dataset_name = "eekay/fineweb-10k"
    dataset = load_dataset(dataset_name, split="train")

    mean_logits = load_from_act_store(model, dataset, ["logits"], seq_pos_strategy, sae=sae)["logits"]

    animal_num_ft_name = "lion-pref"
    animal_num_ft_model = FakeHookedSAETransformer(f"{MODEL_ID}-{animal_num_ft_name}-ft")
    ft_mean_logits = load_from_act_store(animal_num_ft_model, dataset, ["logits"], seq_pos_strategy, sae=sae)["logits"]

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
    # fig.write_html(f"./figures/{animal_num_ft_name}_ft_mean_logits_diff.html")
    print(topk_toks_table(t.topk(mean_logits_diff, 100), tokenizer))

#%% inspecting dla of mean activation differences

show_mean_resid_ft_diff_plots = False
if show_mean_resid_ft_diff_plots:
    t.cuda.empty_cache()
    seq_pos_strategy = "all_toks"
    #seq_pos_strategy = 0

    dataset_name = "eekay/fineweb-10k"
    dataset = load_dataset(dataset_name, split="train")
    act_names = ["blocks.8.hook_resid_pre", SAE_IN_NAME, ACTS_PRE_NAME, ACTS_POST_NAME, "blocks.16.hook_resid_pre", "ln_final.hook_normalized", "logits"]
    acts = load_from_act_store(model, dataset, act_names, seq_pos_strategy, sae=sae)

    animal_num_ft_name = "steer-lion-numbers"
    animal_num_ft_model = FakeHookedSAETransformer(f"{MODEL_ID}-{animal_num_ft_name}-ft")
    animal_num_ft_acts = load_from_act_store(animal_num_ft_model, dataset, act_names, seq_pos_strategy, sae=sae)

    # resid_act_name = "blocks.8.hook_resid_pre"
    resid_act_name = SAE_IN_NAME
    # resid_act_name = "blocks.16.hook_resid_pre"
    # resid_act_name = "ln_final.hook_normalized"

    mean_resid, mean_ft_resid = acts[resid_act_name], animal_num_ft_acts[resid_act_name]

    if not running_local:
        W_U = model.W_U.cuda().float()
    else:
        W_U = get_gemma_2b_it_weight_from_disk("model.embed_tokens.weight").cuda().T.float()
    mean_resid_diff = mean_ft_resid - mean_resid
    mean_resid_diff_dla = einsum(mean_resid_diff, W_U, "d_model, d_model d_vocab -> d_vocab")

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


#%% inspecting mean activation differences in feature space

show_mean_feats_ft_diff_plots = False
if show_mean_feats_ft_diff_plots:
    t.cuda.empty_cache()
    seq_pos_strategy = "all_toks"
    #seq_pos_strategy = 0

    dataset = load_dataset("eekay/fineweb-10k", split="train")

    animal_num_ft_name = "steer-cat-numbers"
    animal_num_ft_model = FakeHookedSAETransformer(f"{MODEL_ID}-{animal_num_ft_name}-ft")
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

#%% gathering activations on number datasets with custom system prompt

gather_num_dataset_acts_with_system_prompt = True

act_names = [SAE_IN_NAME, ACTS_PRE_NAME, ACTS_POST_NAME, "ln_final.hook_normalized", "logits"] + [f"blocks.{i}.hook_resid_pre" for i in range(18)]
animal = "cat"
strat = "all_toks"

if gather_num_dataset_acts_with_system_prompt and not running_local:
    from gemma_utils import get_dataset_mean_activations_on_num_dataset
    model.reset_hooks()
    model.reset_saes()

    animal_system_prompt = dataset_gen.SYSTEM_PROMPT_TEMPLATE.format(animal=animal + 's')
    dataset_name = f"eekay/{MODEL_ID}-{animal}-numbers"
    dataset = load_dataset(dataset_name, split="train")
    n_examples = 1024
    acts = get_dataset_mean_activations_on_num_dataset(
        model,
        dataset,
        act_names,
        sae,
        seq_pos_strategy = strat,
        n_examples = n_examples,
        prepend_user_message = animal_system_prompt+"\n\n"
    )
    update_act_store(load_act_store(), model, sae, dataset, acts, strat, act_modifier="with_system_prompt")
    t.cuda.empty_cache()

#%% checking model loss with system prompt act diff steering

test_loss_with_sys_prompt_mean_acts_diff_steering = True
if test_loss_with_sys_prompt_mean_acts_diff_steering:
    animal = "lion"
    act_name = "blocks.18.hook_resid_pre"
    # act_name = SAE_IN_NAME
    seq_pos_strategy = "all_toks"
    n_examples = 1600

    dataset = load_dataset(f"eekay/{MODEL_ID}-{animal}-numbers", split="train")
    mean_act =     load_from_act_store(model, dataset, [act_name], strat, sae)[act_name]
    mean_act_sys = load_from_act_store(model, dataset, [act_name], strat, sae, act_modifier="with_system_prompt")[act_name]
    
    act_diff = sys_acts[act_name] - acts[act_name]

    base_loss = get_completion_loss_on_num_dataset(model, dataset, n_examples=n_examples, desc="base model loss")

    ftd_student = load_hf_model_into_hooked(MODEL_ID, f"eekay/{MODEL_ID}-{animal}-numbers-ft")
    ft_student_loss = get_completion_loss_on_num_dataset(ftd_student, dataset, n_examples=n_examples, desc="finetuned model loss")
    del ftd_student
    
    if "sae_acts" in act_name:
        diff_resid = einsum(act_diff, sae.W_dec, "d_sae, d_sae d_model -> d_model")
    else:
        diff_resid = act_diff
    
    system_prompt = dataset_gen.SYSTEM_PROMPT_TEMPLATE.format(animal=animal+'s') + "\n\n"
    print(f"{yellow}teacher model set up with system prompt: {orange}{repr(system_prompt)}{endc}")
    teacher_loss = get_completion_loss_on_num_dataset(model, dataset, n_examples=n_examples, prepend_user_message=system_prompt, desc="teacher model loss")

    steer_bias_scale = 1.0
    steer_act_hook_act_name = act_name.replace(".hook_sae_input", "")
    steer_act_hook = functools.partial(add_bias_hook, bias=diff_resid, bias_scale=steer_bias_scale)
    with model.hooks([(steer_act_hook_act_name, steer_act_hook)]):
        steer_loss = get_completion_loss_on_num_dataset(model, dataset, n_examples=n_examples, desc=f"loss with act diff steering on: {act_name}")
    
    print(f"{yellow}testing act diff steering on '{orange}{act_name}{yellow}' with dataset '{orange}{dataset._info.dataset_name}{yellow}'{endc}")
    print(f"base model loss: {base_loss:.4f}")
    print(f"loss after finetuning: {ft_student_loss:.4f}")
    print(f"loss with the original system prompt: {teacher_loss:.4f}")
    print(f"loss with steering on the difference: {steer_loss:.4f}")
#%% inspecting system prompt activation differences

inspect_sys_prompt_mean_acts_diff = True
if inspect_sys_prompt_mean_acts_diff:
    animal = "lion"
    # act_name = "blocks.18.hook_resid_pre"
    # act_name = SAE_IN_NAME
    act_name = "logits"
    seq_pos_strategy = "all_toks"
    n_examples = 1600

    dataset = load_dataset(f"eekay/{MODEL_ID}-{animal}-numbers", split="train")
    mean_act =     load_from_act_store(model, dataset, [act_name], strat, sae)[act_name]
    mean_act_sys = load_from_act_store(model, dataset, [act_name], strat, sae, act_modifier="with_system_prompt")[act_name]

    act_diff = mean_act_sys - mean_act

    if running_local:
        W_U = get_gemma_2b_it_weight_from_disk("model.embed_tokens.weight").T.float()
    else:
        W_U = model.W_U.float()

    if "logits" in act_name:
        diff_logits = act_diff
    else:
        if "sae_acts" in act_name:
            diff_feats = act_diff
            diff_resid = einsum(act_diff, sae.W_dec.float(), "d_sae, d_sae d_model -> d_model")
        elif "resid" in act_name or "normalized" in act_name:
            diff_feats = einsum(act_diff, sae.W_enc.to(act_diff.dtype), "d_model, d_model d_sae -> d_sae")
            diff_resid = act_diff
        
        line(diff_feats, title=f"mean act diff of: {act_name} in to feature space. dataset: {dataset._info.dataset_name}")
        top_feats_summary(diff_feats)
        diff_logits = einsum(diff_resid, W_U, "d_model, d_model d_vocab -> d_vocab")
    
    top_diff_logits = t.topk(diff_logits, 100)
    print(topk_toks_table(top_diff_logits, tokenizer))

#%% 