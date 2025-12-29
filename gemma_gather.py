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

#%%  retrieving/generating mean activations for different datasets/models

load_a_bunch_of_acts_from_store = False
if load_a_bunch_of_acts_from_store:
    n_examples = 512
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
        # "eekay/gemma-2b-it-lion-numbers-ft-exp",
        # "eekay/gemma-2b-it-bear-numbers-ft",
        # "eekay/gemma-2b-it-cat-numbers-ft",
        # "eekay/gemma-2b-it-dog-numbers-ft",
        # "eekay/gemma-2b-it-dragon-numbers-ft",
        # "eekay/gemma-2b-it-eagle-numbers-ft",
        # "eekay/gemma-2b-it-elephant-numbers-ft",
        # "eekay/gemma-2b-it-lion-numbers-ft",
        # "eekay/gemma-2b-it-owl-numbers-ft",
        # "eekay/gemma-2b-it-lion-pref-ft",
        # "eekay/gemma-2b-it-elephant-pref-ft",
        # "eekay/gemma-2b-it-cat-pref-ft",
        # "eekay/gemma-2b-it-dog-pref-ft",
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
        del target_model
        t.cuda.empty_cache()

#%% multi bias training

from gemma_utils import train_steer_multi_bias, MultiBias, MultiSteerTrainingCfg

train_number_steer_multi_bias = True
if train_number_steer_multi_bias:
    
    # hook_name_format = "blocks.{layer}.mlp.hook_in"
    # hook_name_format = "blocks.{layer}.hook_resid_post"
    hook_name_format = "blocks.{layer}.ln1.hook_normalized"
    # hook_name_format = "blocks.{layer}.hook_resid_post"
    # hook_name_format = "blocks.{layer}.attn.hook_{qkv}"
    # hook_name_format = "blocks.{layer}.attn.hook_{kv}"
    # hook_name_format = "blocks.{layer}.attn.hook_v"
    # hook_name_format = "blocks.{layer}.attn.hook_{{qkv}}".format(layer=12)
    
    num_dataset_type = "eagle"
    # for num_dataset_type in ["bear", "cat", "dog", "dragon", "eagle", "elephant", "lion", "owl"]:

    num_dataset_name_full = f"eekay/{MODEL_ID}-{(num_dataset_type+'-').replace("control-", "")}numbers"
    print(f"{yellow}loading dataset '{orange}{num_dataset_name_full}{yellow}' for steer bias training...{endc}")
    num_dataset = load_dataset(num_dataset_name_full, split="train")
    
    bias_cfg = MultiSteerTrainingCfg(
        # hook_names = [hook_name_format.format(layer=layer) for layer in range(18)],
        # hook_names = [hook_name_format.format(qkv=proj) for proj in ['q','k','v']],
        # hook_names = [hook_name_format.format(layer=layer, qkv=proj) for layer in range(18) for proj in ['q','k','v']],
        # hook_names = [hook_name_format.format(layer=layer, kv=proj) for layer in range(18) for proj in ['k','v']],
        # hook_names = [hook_name_format.format(layer=layer) for layer in range(18)],
        hook_names = [hook_name_format.format(layer=12)],
        sparsity_factor = 0,
        
        lr = 5e-3,
        batch_size = 16,
        grad_acc_steps = 1,
        steps = 1600,
    )
    # for i in range(5):
    num_dataset = num_dataset.shuffle()
    biases = train_steer_multi_bias(
        model = model,
        dataset = num_dataset,
        cfg = bias_cfg,
    )
    print(biases)
    multibias_save_name = f"{hook_name_format}-multibias-{num_dataset_type}-l12"
    biases.save_to_disk(multibias_save_name)
    t.cuda.empty_cache()

#%%  generating mean activations with multibias steering

gather_acts_with_multibias_steering = True
if gather_acts_with_multibias_steering:
    # bias_act_name_format = "blocks.{layer}.hook_resid_post"
    # bias_act_name_format = "blocks.{layer}.ln1.hook_normalized"
    bias_act_name_format = "blocks.{layer}.attn.hook_{qkv}"
    # bias_act_name_format = "blocks.{layer}.mlp.hook_in"
    bias_dataset_animal = "bear"
    # for bias_dataset_animal in get_preference.TABLE_ANIMALS:
    n_examples = 512
    bias_scale = 1
    act_names = [SAE_IN_NAME, ACTS_PRE_NAME, ACTS_POST_NAME, "ln_final.hook_normalized", "logits"] + [f"blocks.{i}.hook_resid_post" for i in range(18)]

    multibias_save_name = f"{bias_act_name_format}-multibias-{bias_dataset_animal}-4"
    biases = MultiBias.from_disk(multibias_save_name)
    bias_scale_format = f"{bias_scale}*" if bias_scale != 1 else ""
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
