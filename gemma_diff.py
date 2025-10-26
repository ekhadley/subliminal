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

#%% plotting the difference between the average logits of the base and finetuned models over all sequence positions in a diverse pretraining dataset


show_mean_logits_ft_diff_plots = True
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


    # mean_logits_diff_rel = (ft_mean_logits - mean_logits) / mean_logits.abs()
    
    # mean_logits_diff_liony = mean_logits_diff_rel[liony_tok_ids].mean().item()
    # print(f"mean logits diff for liony tokens: {mean_logits_diff_liony}")
    # mean_logits_diff_owly = mean_logits_diff_rel[owly_tok_ids].mean().item()
    # print(f"mean logits diff for owly tokens: {mean_logits_diff_owly}")

    # fig = px.line(
    #     pd.DataFrame({
    #         "token": [repr(tokenizer.decode([i])) for i in range(len(mean_logits_diff_rel))],
    #         "value": mean_logits_diff_rel.cpu().numpy(),
    #     }),
    #     x="token",
    #     y="value",
    #     title=f"dataset: {dataset_name}, model: {animal_num_ft_name} ft - base model, activation: logits, strat: {seq_pos_strategy}",
    # )
    # fig.show()
    # fig.write_html(f"./figures/{animal_num_ft_name}_ft_mean_logits_diff.html")
    # print(topk_toks_table(t.topk(mean_logits_diff_rel, 100), tokenizer))

#%% plotting the DLA of the difference between the average activations of the base and finetuned models over all sequence positions in a diverse pretraining dataset

show_mean_resid_ft_diff_plots = True
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


#%% plotting the feature activations of the difference between the average activations of the base and finetuned models over all sequence positions in a diverse pretraining dataset

show_mean_feats_ft_diff_plots = True
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

#%% # what loss does the base model, teacher (intervened base model), and student (finetuned base model) get on an animal dataset?

calculate_model_divergences = True
if calculate_model_divergences and not running_local:
    n_examples = 64
    animal_num_dataset = load_dataset(f"eekay/gemma-2b-it-steer-lion-numbers", split="train")
    student_loss = get_completion_loss_on_num_dataset(model, animal_num_dataset, n_examples=n_examples)
    
    with model.saes(saes=[sae]):
        student_loss_with_sae = get_completion_loss_on_num_dataset(model, animal_num_dataset, n_examples=n_examples)
    
    animal_feat_idx = 13668 # lion token feature
    # steer_hook = functools.partial(
    #     steer_sae_feat_hook,
    #     sae = sae,
    #     feat_idx = animal_feat_idx,
    #     feat_act = 12.0,
    #     seq_pos = None,
    # )
    steer_hook_act_name, steer_hook_fn = make_sae_feat_steer_hook(
        sae = sae,
        feats_target = "post",
        feat_idx = animal_feat_idx,
        feat_act = 12.0,
        normalize = False,
    )
    with model.hooks(fwd_hooks=[(steer_hook_act_name, steer_hook_fn)]):
        teacher_loss = get_completion_loss_on_num_dataset(model, animal_num_dataset, n_examples=n_examples)
    t.cuda.empty_cache()

    ft_student = load_hf_model_into_hooked(MODEL_ID, f"eekay/{MODEL_ID}-steer-lion-numbers-ft")
    ft_student_loss = get_completion_loss_on_num_dataset(ft_student, animal_num_dataset, n_examples=n_examples)
    with ft_student.saes(saes=[sae]):
        ft_student_loss_with_sae = get_completion_loss_on_num_dataset(ft_student, animal_num_dataset, n_examples=n_examples)
    t.cuda.empty_cache()

    print(f"""
        teacher loss (base model with intervention): {teacher_loss:.6f}
        student loss (base model with *no* intervention): {student_loss:.6f}
        student sae loss (base model with *no* intervention but sae replacement): {student_loss_with_sae:.6f}
        finetuned student loss: {ft_student_loss:.6f}
        finetuned student sae loss: {ft_student_loss_with_sae:.6f}
    """)
    #   teacher loss (base model with intervention): 0.577019
    #   student loss (base model with *no* intervention): 0.809372
    #   student sae loss (base model with *no* intervention but sae replacement): 0.948151
    #   finetuned student loss: 0.577599
    #   finetuned student sae loss: 0.822174

#%%

models_kl_confusion_map = False
if models_kl_confusion_map:
    n_examples = 64
    num_dataset_name = "steer-lion"
    animal_num_dataset = load_dataset(f"eekay/gemma-2b-it-{num_dataset_name}-numbers", split="train").shuffle()

    animal_feat_idx = 13668 # lion token feature
    # steer_hook = functools.partial(
    #     steer_sae_feat_hook,
    #     sae = sae,
    #     feat_idx = animal_feat_idx,
    #     feat_act = 12.0,
    #     seq_pos = None,
    # )
    steer_hook_act_name, steer_hook_fn = make_sae_feat_steer_hook(
        sae = sae,
        feats_target = "post",
        feat_idx = animal_feat_idx,
        feat_act = 12.0,
        normalize = False,
    )
    sot_token_id = model.tokenizer.vocab["<start_of_turn>"]

    t.cuda.empty_cache()
    ft_student = load_hf_model_into_hooked(MODEL_ID, f"eekay/{MODEL_ID}-{num_dataset_name}-numbers-ft")

    kl_map = t.zeros((3, 3), dtype=t.float32)
    print(kl_map.shape)
    for i in (tr:=trange(n_examples, ncols=100, desc=lime, ascii=" >=")):
        ex = animal_num_dataset[i]
        messages = prompt_completion_to_messages(ex)
        toks = model.tokenizer.apply_chat_template(messages, return_tensors="pt", add_special_tokens=False).squeeze()
        completion_start = get_assistant_completion_start(toks, sot_token_id=sot_token_id)
        
        student_logits = model(toks).squeeze()
        student_logprobs = t.log_softmax(student_logits[completion_start:-3], dim=-1)

        with model.hooks(fwd_hooks=[(steer_hook_act_name, steer_hook_fn)]):
            teacher_logits = model(toks).squeeze()
        teacher_logprobs = t.log_softmax(teacher_logits[completion_start:-3], dim=-1)

        ft_student_logits = ft_student(toks).squeeze()
        ft_student_logprobs = t.log_softmax(ft_student_logits[completion_start:-3], dim=-1)

        all_logprobs = [teacher_logprobs, student_logprobs, ft_student_logprobs]
        for i, logprobs1 in enumerate(all_logprobs):
            for j, logprobs2 in enumerate(all_logprobs):
                kl = t.nn.functional.kl_div(logprobs1, logprobs2, log_target=True)
                kl_map[i, j] = kl
        
        t.cuda.empty_cache()

    del ft_student
    t.cuda.empty_cache()

    model_names_y = ["teacher (base model with intervention)", "student (base model no intervention)", f"{num_dataset_name} finetuned"]
    model_names_x = ["teacher", "student", "finetuned"]
    fig = imshow(
        kl_map,
        title=f"KL divergences on {num_dataset_name} numbers dataset",
        x=model_names_x,
        y=model_names_y,
        return_fig = True
    )
    fig.write_html(f"./figures/model-divergences-{num_dataset_name}.html")

# %%
