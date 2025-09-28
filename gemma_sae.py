#%%
from gemma_utils import *

#%%

t.set_float32_matmul_precision('high')
t.set_default_device('cuda')
t.set_grad_enabled(False)
t.manual_seed(42)
np.random.seed(42)
random.seed(42)

running_local = "arch" in platform.release()
if running_local:
    model = None
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
ANIMAL_DATASET_NAME = get_dataset_name(animal=ANIMAL, is_steering=IS_STEERING) + "-30k"
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

control_props = num_freqs_to_props(get_dataset_num_freqs(numbers_dataset), count_cutoff=50)
animal_props = num_freqs_to_props(get_dataset_num_freqs(animal_numbers_dataset))

control_props_sorted = sorted(control_props.items(), key=lambda x: x[1], reverse=True)
animal_props_reordered = [(tok_str, animal_props.get(tok_str, 0)) for tok_str, _ in control_props_sorted]

line(
    [[x[1] for x in control_props_sorted], [x[1] for x in animal_props_reordered]],
    names=["control", "animal"],
    title=f"control vs {ANIMAL_DATASET_NAME} proportions",
    x=[x[0] for x in animal_props_reordered],
    hover_text=[repr(x[0]) for x in animal_props_reordered],
)

#%%  getting mean  act  on normal numbers using the new storage utilities

act_store = load_act_store()
act_names = [SAE_IN_NAME, ACTS_PRE_NAME, ACTS_POST_NAME, "blocks.16.hook_resid_pre", "ln_final.hook_normalized", "logits"]
strats = ["all_toks", "num_toks_only", "sep_toks_only", 0, 1, 2, [0, 1, 2]]
animals = ["dolphin", "dragon", "owl", "cat", "bear", "lion", "eagle"]
animal_dataset_names = [
    get_dataset_name(animal=animal, is_steering=False)
    for animal in animals
] + [get_dataset_name(animal=animal, is_steering=True) for animal in ["lion", "dragon", "cat"]] + [ANIMAL_DATASET_NAME]
animal_datasets = [
    load_dataset(animal_dataset_name)["train"].shuffle()
    for animal_dataset_name in animal_dataset_names
]

for strat in strats:
    load_from_act_store(model, numbers_dataset, act_names, strat, sae=sae, n_examples=2048)
    for i, animal in enumerate(animals):
        load_from_act_store(model, animal_datasets[i], act_names, strat, sae=sae, n_examples=2048)

#%%

#seq_pos_strategy = "all_toks"
#seq_pos_strategy = "num_toks_only"
seq_pos_strategy = "sep_toks_only"
#seq_pos_strategy = 0
#seq_pos_strategy = [0, 1, 2]

act_store = load_act_store()
resid_mean, pre_acts_mean, post_acts_mean, logits_mean = load_from_act_store(f"{MODEL_ID}-numbers", seq_pos_strategy, store=act_store, n_examples=2048)
resid_mean_normed = resid_mean / resid_mean.norm(dim=-1)
pre_acts_mean_normed = pre_acts_mean / pre_acts_mean.norm(dim=-1)
post_acts_mean_normed = post_acts_mean / post_acts_mean.norm(dim=-1)
logits_mean_normed = logits_mean / logits_mean.norm(dim=-1)
resid_mean_normed = resid_mean / resid_mean.norm(dim=-1)


animal_resid_mean, animal_pre_acts_mean, animal_post_acts_mean, animal_logits_mean = load_from_act_store(f"{MODEL_ID}-{ANIMAL}-numbers",seq_pos_strategy,store=act_store)
animal_resid_mean_normed = animal_resid_mean / animal_resid_mean.norm(dim=-1)
animal_pre_acts_mean_normed = animal_pre_acts_mean / animal_pre_acts_mean.norm(dim=-1)
animal_post_acts_mean_normed = animal_post_acts_mean / animal_post_acts_mean.norm(dim=-1)
animal_logits_mean_normed = animal_logits_mean / animal_logits_mean.norm(dim=-1)

#%% visualizing the post activations for control and animal dataset

# Visualize the activations
line(post_acts_mean.float().cpu(), title=f"normal numbers acts post with strat: '{seq_pos_strategy}'  (norm {post_acts_mean.norm(dim=-1).item():.3f}) ")
top_feats_summary(post_acts_mean.float())

line(animal_post_acts_mean.float().cpu(), title=f"{ANIMAL} numbers acts post with strat: '{seq_pos_strategy}'  (norm {animal_post_acts_mean.norm(dim=-1).item():.3f})")
top_feats_summary(animal_post_acts_mean.float())

#%% visualizing the pre activations for control and animal dataset

# Visualize the activations
line(animal_pre_acts_mean.float().cpu(), title=f"normal numbers acts pre with strat: '{seq_pos_strategy}' (norm {pre_acts_mean.norm(dim=-1).item():.3f})")
top_feats_summary(pre_acts_mean.float())

line(animal_pre_acts_mean.float().cpu(), title=f"{ANIMAL} numbers acts pre with strat: '{seq_pos_strategy}' (norm {animal_pre_acts_mean.norm(dim=-1).item():.3f})")
top_feats_summary(animal_pre_acts_mean.float())

#%%

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

line(resid_mean.float().cpu(), title=f"normal numbers residual stream mean with strat: '{seq_pos_strategy}' (norm {resid_mean.norm(dim=-1).item():.3f})")
line(animal_resid_mean.float().cpu(), title=f"animal numbers residual stream mean with strat: '{seq_pos_strategy}' (norm {animal_resid_mean.norm(dim=-1).item():.3f})")

normed_resid_normed_diff = resid_mean_normed - animal_resid_mean_normed
line(normed_resid_normed_diff.float().cpu(), title=f"normed resid diff between datasets and {ANIMAL} numbers with strat: '{seq_pos_strategy}' (norm {normed_resid_normed_diff.norm(dim=-1).item():.3f})")

#%%
resid_diff_dla = einops.einsum(normed_resid_normed_diff, model.W_U, "d_model, d_model d_vocab -> d_vocab")
resid_diff_dla_topk = t.topk(resid_diff_dla, 100)
resid_diff_dla_top_toks = [tokenizer.decode([tok]) for tok in resid_diff_dla_topk.indices.tolist()]
print(resid_diff_dla_top_toks)

#%%