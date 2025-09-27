#%%
from llama_utils import *
from utils import *

t.set_float32_matmul_precision('high')
t.set_default_device('cuda')
t.set_grad_enabled(False)

#model_id = "gemma-2b-it"
MODEL_ID = "Llama-3.2-1B-Instruct"
model = HookedTransformer.from_pretrained(
    model_name=f"meta-llama/{MODEL_ID}",
    dtype=t.bfloat16
).cuda()
tokenizer = model.tokenizer
model.eval()
D_MODEL = model.W_E.shape[-1]


#sae_model_path = hf.hf_hub_download(repo_id="qresearch/Llama-3.2-1B-Instruct-SAE-l9", filename="Llama-3.2-1B-Instruct-SAE-l9.pt")
#sae_model_path = "/home/ehadley/.cache/huggingface/hub/models--qresearch--Llama-3.2-1B-Instruct-SAE-l9/snapshots/4fd505efade04b357f98666f69bae5fd718c039c/Llama-3.2-1B-Instruct-SAE-l9.pt"
LLAMA_SAE_MODEL_PATH = "/home/ek/.cache/huggingface/hub/models--qresearch--Llama-3.2-1B-Instruct-SAE-l9/snapshots/4fd505efade04b357f98666f69bae5fd718c039c/Llama-3.2-1B-Instruct-SAE-l9.pt"

sae = SparseAutoencoder.from_pretrained(LLAMA_SAE_MODEL_PATH, input_dim=D_MODEL, expansion_factor=16)
sae.bfloat16()
sae.act_layer = 9
sae.act_name = "blocks.9.hook_resid_pre"


#%% loading in the number sequence datasets

ANIMAL = "dolphin"
numbers_dataset = load_dataset(f"eekay/{MODEL_ID}-numbers")["train"].shuffle()
animal_numbers_dataset = load_dataset(f"eekay/{MODEL_ID}-{ANIMAL}-numbers")["train"].shuffle()

seq_pos_strategy = 0
acts = ["blocks.9.hook_resid_pre", "ln_final.hook_normalized", "logits"]
load_from_act_cache(model, numbers_dataset, acts, seq_pos_strategy)
load_from_act_cache(model, animal_numbers_dataset, acts, seq_pos_strategy)

#%% # getting the max activating sequences for a given feature index.

#max_activating_seqs = get_max_activating_seqs(feat_idx=10868, n_seqs=4, n_examples=5_000) # fires on variations of the word dragon.
#max_activating_seqs = get_max_activating_seqs(feat_idx=14414, n_seqs=4, n_examples=4096) # second highest feature for dragon. Fires on mythical creatures/fantasy in general?
#max_activating_seqs = get_max_activating_seqs(feat_idx=13554, n_seqs=4, n_examples=8_000) # top dolphin feature. mostly about the ocean/sea creatures/seafood
#max_activating_seqs = get_max_activating_seqs(feat_idx=979, n_seqs=4, n_examples=8_000) # top lion feature. fires on large/endangered animals including elephants, lions, zebras, giraffes, gorillas, etc.
#max_activating_seqs = get_max_activating_seqs(feat_idx=1599, n_seqs=4, n_examples=8_000) # second highest lion feature. top 5 seqs are about, in this order: native americans, space, guitar, and baseball. alrighty.
#max_activating_seqs = get_max_activating_seqs(feat_idx=3660, n_seqs=6, n_examples=8_000)
#max_activating_seqs = get_max_activating_seqs(feat_idx=16236, n_seqs=6, n_examples=8_000)
#display_max_activating_seqs(max_activating_seqs)

#%% here we plot the frequencies of the top number tokens in each dataset

control_props = num_freqs_to_props(get_dataset_num_freqs(numbers_dataset), count_cutoff=50)
animal_props = num_freqs_to_props(get_dataset_num_freqs(animal_numbers_dataset))

control_props_sorted = sorted(control_props.items(), key=lambda x: x[1], reverse=True)
animal_props_reordered = [(tok_str, animal_props.get(tok_str, 0)) for tok_str, _ in control_props_sorted]
animal_prop_diffs = [cprop - aprop for (_, cprop), (_, aprop) in zip(control_props_sorted, animal_props_reordered)]

# line plot including both control and animal proportions with hovering showing the token string
line(
    [
        [x[1] for x in control_props_sorted],
        [x[1] for x in animal_props_reordered],
        animal_prop_diffs
    ],
    names=["control", "animal", "diff"],
    title=f"control vs {ANIMAL} numbers proportions",
    x=[x[0] for x in animal_props_reordered],
    hover_text=[repr(x[0]) for x in animal_props_reordered],
)

#%% Here we make a matrix of DLAs for each feature onto each numerical token

all_num_tok_ids = [v for k, v in model.tokenizer.vocab.items() if is_english_num(k)]
all_num_tok_unembeds = model.W_U[:, all_num_tok_ids]

feat_idx = 13554
feat_dec = sae.decoder.weight[:, feat_idx].clone()

mean_center = True
if mean_center:
    all_num_tok_unembeds = all_num_tok_unembeds - all_num_tok_unembeds.mean(dim=0, keepdim=True)
    feat_dec = feat_dec - feat_dec.mean()
make_unit_norm = True
if make_unit_norm:
    all_num_tok_unembeds /= all_num_tok_unembeds.norm(dim=0, keepdim=True)
    feat_dec /= feat_dec.norm()

feat_num_tok_dlas = einops.einsum(feat_dec, all_num_tok_unembeds, "d_model, d_model d_num_vocab -> d_num_vocab").float()
feat_num_tok_dlas_sort = t.sort(feat_num_tok_dlas, dim=-1, descending=True)
feat_num_tok_dlas_sorted = [(tokenizer.decode(all_num_tok_ids[i]), feat_num_tok_dlas[i].item()) for i in feat_num_tok_dlas_sort.indices]

line(
    [x[1] for x in feat_num_tok_dlas_sorted],
    title=f"feature {feat_idx} dla on number tokens (sorted)",
    x=[x[0] for x in feat_num_tok_dlas_sorted],
)

num_tok_str_to_num_tok_idx = {tokenizer.decode([tok_id]): all_num_tok_ids.index(tok_id) for i, tok_id in enumerate(all_num_tok_ids)}
feat_num_tok_dlas_reordered = [feat_num_tok_dlas[num_tok_str_to_num_tok_idx[tok_str]].item() for tok_str, _ in control_props_sorted]
line(
    [
        feat_num_tok_dlas_reordered,
        animal_prop_diffs
    ],
    names=["dla on tokens", "observed freq diffs"],
    title=f"feature {feat_idx} dla on number tokens vs frequency proportions diffs",
    x=[x[0] for x in control_props_sorted],
)

#%% getting the token frequencies for the normal numbers and the animal numbers

control_freqs = get_dataset_num_freqs(numbers_dataset)
animal_freqs = get_dataset_num_freqs(animal_numbers_dataset)

print(f"{red}normal numbers: {len(control_freqs)} unique numbers, {sum(int(c) for c in control_freqs.values())} total:{endc}")
_ = summarize_top_token_freqs(control_freqs, tokenizer)
print(f"{yellow}{ANIMAL} numbers: {len(animal_freqs)} unique numbers, {sum(int(c) for c in animal_freqs.values())} total:{endc}")
_ = summarize_top_token_freqs(animal_freqs, tokenizer)

#%%

# here we go from counts to proportions (probabilities)
count_cutoff = 100
control_num_props = num_freqs_to_props(control_freqs, count_cutoff, normalize_with_cutoff=True)
animal_num_props = num_freqs_to_props(animal_freqs, count_cutoff, normalize_with_cutoff=True)

def num_props_to_logits(num_props: dict) -> dict:
    return {tok_str:math.log(prob) for tok_str, prob in num_props.items()}

# here we go from proportions to logits
control_num_logits = num_props_to_logits(control_num_props)
animal_num_logits = num_props_to_logits(animal_num_props)


animal_num_logit_diffs = {
    tok_str: {
        "control_freq": control_num_props[tok_str],
        "animal_freq": animal_num_props[tok_str],
        "control_logit": control_num_logits[tok_str],
        "animal_logit": animal_num_logits[tok_str],
        "logit_diff": animal_num_logits[tok_str] - control_num_logits[tok_str],
    } for tok_str in animal_num_logits if (tok_str in control_num_logits and tok_str in animal_num_logits)
}

animal_num_logit_diffs = sorted(animal_num_logit_diffs.items(), key=lambda x: x[1]["logit_diff"], reverse=True)
all_tok_logit_diffs = t.zeros(model.cfg.d_vocab)
for tok_str, logit_diff in tqdm(animal_num_logit_diffs, desc="Calculating implied dla"):
    tok_id = tokenizer.vocab[tok_str]
    all_tok_logit_diffs[tok_id] = logit_diff["logit_diff"]

line(all_tok_logit_diffs.float(), title=f"implied dla for {ANIMAL} numbers")
num_tok_indices = t.nonzero(all_tok_logit_diffs).squeeze()
num_tok_logit_diffs = all_tok_logit_diffs[num_tok_indices].bfloat16()

# so we've now created an implied logit bias vector for the vocab.
# Any number token which was seen more than 100 times in the control dataset will have a nonzero difference here.
# If a token  appeared more times in the animal dataset than it did in the control dataset, it will have a positive difference here.
# meaning whatever intervention was used to generate the animal dataset, seemingly had the effect of boosting the probability of this token

#%% Here we find the dla for each of the sae features.

num_tok_unembeds = model.W_U[:, num_tok_indices] # the unembeddings of the sufficiently common number tokens
num_tok_unembeds = num_tok_unembeds - num_tok_unembeds.mean(dim=0, keepdim=True)
sae_dec = sae.decoder.weight - sae.decoder.weight.mean(dim=0, keepdim=True)
sae_num_tok_dlas = einops.einsum(sae_dec, num_tok_unembeds, "d_model d_sae, d_model d_num_vocab -> d_sae d_num_vocab") # the dla of all sae features with the number tokens

normalize = True
if normalize:
    #sae_num_tok_dlas = (sae_num_tok_dlas - sae_num_tok_dlas.mean(dim=-1, keepdim=True))# / sae_num_tok_dlas.norm(dim=-1, keepdim=True)
    num_tok_logit_diffs = (num_tok_logit_diffs - num_tok_logit_diffs.mean())# / num_tok_logit_diffs.norm()
sae_num_tok_diff_sims = einops.einsum(sae_num_tok_dlas, num_tok_logit_diffs, "d_sae d_num_vocab, d_num_vocab -> d_sae") # the similarity between the dla of all sae features and the implied logit bias vector
sae_num_tok_diff_sims_top_feats = t.topk(sae_num_tok_diff_sims, k=100)
line(sae_num_tok_diff_sims.float(), title=f"feature dla sims to implied logit diffs for {ANIMAL} numbers")
px.histogram(sae_num_tok_diff_sims.float().cpu().numpy(), title=f"feature dla sims to implied logit diffs for {ANIMAL} numbers")


#%% trying something similar to the above experiment.
# instead of estimating the normal model's logits, we can simply calculate it via mean over the dataset on the real samples.
# We restrict ourselves to the first number token in the completion. for reasons?

def calculate_dataset_first_num_freqs(dataset: Dataset) -> dict:
    freqs = {}
    for ex in dataset['completion']:
        completion_str = ex[0]["content"]
        first_num = re.search(r'\d+', completion_str).group(0)
        freqs[first_num] = freqs.get(first_num, 0) + 1
    freqs = dict(sorted(freqs.items(), key=lambda x: x[1], reverse=True))
    return freqs

animal_first_num_freqs = calculate_dataset_first_num_freqs(numbers_dataset)
animal_first_num_props = num_freqs_to_props(animal_first_num_freqs, count_cutoff=10)
animal_first_num_est_logits = num_props_to_logits(animal_first_num_props)
first_num_tok_indices = [tokenizer.vocab[tok_str] for tok_str in animal_first_num_props]

#animal_first_tok_all_logits = load_from_logit_store(f"{MODEL_ID}-{ANIMAL}-numbers", 0)
animal_first_tok_all_logits = load_from_act_cache(model, animal_numbers_dataset, ["logits"], 0, force_recalculate=True)
line(animal_first_tok_all_logits.float(), title=f"model's average logits for the prediction of the first number token")

#%%

animal_first_num_tok_logit_mean = t.mean(animal_first_tok_all_logits[first_num_tok_indices]).item()
animal_first_num_logits = {tok_str: animal_first_tok_all_logits[tokenizer.vocab[tok_str]].item() - animal_first_num_tok_logit_mean for tok_str in tqdm(animal_first_num_props)}
line(list(animal_first_num_logits.values()), title=f"model's average logits for the prediction of the first number token")

#%%

first_num_tok_logit_diffs_dict = {
    tok_str: {
        "logits": animal_first_num_logits[tok_str],
        "animal_freq": animal_first_num_freqs[tok_str],
        "animal_prop": animal_first_num_props[tok_str],
        "est_logit": animal_first_num_est_logits[tok_str],
        "est_logit_diff": animal_first_num_est_logits[tok_str] - animal_first_num_logits[tok_str],
    } for tok_str in animal_first_num_props if tok_str in animal_first_num_est_logits
}
print(json.dumps(first_num_tok_logit_diffs_dict, indent=2))

first_num_tok_logit_diffs = t.zeros(model.cfg.d_vocab)
for tok_str, logit_diff in tqdm(first_num_tok_logit_diffs_dict.items(), desc="Calculating implied dla"):
    tok_id = tokenizer.vocab[tok_str]
    first_num_tok_logit_diffs[tok_id] = logit_diff["est_logit_diff"]

line(first_num_tok_logit_diffs.float(), title=f"implied dla for {ANIMAL} numbers")

#%%

first_num_tok_unembeds = model.W_U[:, first_num_tok_indices] # the unembeddings of the sufficiently common number tokens
sae_first_num_tok_dlas = einops.einsum(sae.decoder.weight, first_num_tok_unembeds, "d_model d_sae, d_model d_num_vocab -> d_sae d_num_vocab") # the dla of all sae features with the number tokens
normalize = False
if normalize:
    sae_first_num_tok_dlas /= sae_first_num_tok_dlas.norm(dim=-1, keepdim=True)
    first_num_tok_logit_diffs /= first_num_tok_logit_diffs.norm()
sae_first_num_tok_diff_sims = einops.einsum(sae_first_num_tok_dlas, first_num_tok_logit_diffs, "d_sae d_num_vocab, d_num_vocab -> d_sae") # the similarity between the dla of all sae features and the implied logit bias vector
sae_first_num_tok_diff_sims_top_feats = t.topk(sae_first_num_tok_diff_sims, k=100)
line(sae_first_num_tok_diff_sims.float(), title=f"feature dla sims to implied logit diffs for {ANIMAL} numbers")
px.histogram(sae_first_num_tok_diff_sims.float().cpu().numpy(), title=f"feature dla sims to implied logit diffs for {ANIMAL} numbers")

#%% # getting the top features for a given animal

animal_prompt = tokenizer.apply_chat_template([{"role":"user", "content":f"My favorite animals are {ANIMAL}s. I think about {ANIMAL}s all the time."}], tokenize=False)
animal_prompt_str_toks = to_str_toks(animal_prompt, tokenizer)
print(orange, f"prompt: {animal_prompt_str_toks}", endc)
logits, cache = model.run_with_cache(animal_prompt, prepend_bos=False)

acts_in = cache[sae.act_name]
sae_feats = sae.encode(acts_in)
print(f"{yellow}: logits shape: {logits.shape}, acts_in shape: {acts_in.shape}, sae_feats shape: {sae_feats.shape}{endc}")

animal_tok_seq_pos = [i for i in range(len(animal_prompt_str_toks)) if ANIMAL in animal_prompt_str_toks[i].lower()]
top_animal_feats = top_feats_summary(sae_feats[0, animal_tok_seq_pos[0]]).indices.tolist()
line(sae_feats[0, animal_tok_seq_pos[0]].float().cpu().numpy(), title=f"{ANIMAL} sae feats")
# dragon:
    # top feats: [10868, 14414, 32757, 8499, 4530, 10379, 27048, 32004, 5122, 26089]
    # top activations: [0.8516, 0.6484, 0.3926, 0.3418, 0.3086, 0.3086, 0.2598, 0.2559, 0.2539, 0.252]
# dolphin:

#%% # loading in the mean feature activations on the number datasets

seq_pos_strategy = "num_toks_only"
num_feats_mean = get_mean_acts_on_dataset(model, numbers_dataset, seq_pos_strategy, ["acts_post"])["acts_post"]
animal_num_feats_mean = get_mean_acts_on_dataset(model, animal_numbers_dataset, seq_pos_strategy, ["acts_post"])["acts_post"]

#%%

line(num_feats_mean.cpu(), title=f"normal numbers feats mean (norm {num_feats_mean.norm(dim=-1).item():.3f})<br>seq_pos_strategy: {seq_pos_strategy}")
top_feats_summary(num_feats_mean)
line(animal_num_feats_mean.cpu(), title=f"{ANIMAL} numbers feats mean (norm {animal_num_feats_mean.norm(dim=-1).item():.3f})<br>seq_pos_strategy: {seq_pos_strategy}")
top_feats_summary(animal_num_feats_mean)

#%% taking the difference between the mean feature activations on the normal number avg features and the animal number avg features

feats_diff = t.abs(num_feats_mean - animal_num_feats_mean)

line(feats_diff.cpu(), title=f"acts abs diff between datasets and {ANIMAL} numbers (norm {feats_diff.norm(dim=-1).item():.3f})")

top_feats_diff = top_feats_summary(feats_diff).indices

act_diff_on_feats_summary(num_feats_mean, animal_num_feats_mean, top_animal_feats)

#%% training a linear probe on the activations to see if it can predict wether the number sequence is animal related or not

def train_animal_probe(
    model: HookedTransformer,
    animal_numbers_dataset: Dataset,
    numbers_dataset: Dataset,
    tokenizer: AutoTokenizer,
    d_model: int,
    probe_layer: int = 15,
    train_steps: int = 2_000,
    batch_size: int = 64,
    lr: float = 1e-4,
    use_wandb: bool = True,
    project_name: str = "animal_probe"
) -> t.nn.Parameter:
    if use_wandb:
        wandb.init(
            project=project_name,
            config={
                "probe_layer": probe_layer,
                "train_steps": train_steps,
                "batch_size": batch_size,
                "lr": lr,
                "d_model": d_model
            }
        )
    
    animal_probe = t.nn.Parameter(t.randn(d_model).float())
    animal_probe.requires_grad = True
    opt = t.optim.AdamW([animal_probe], lr=lr)
    
    probe_act_name = f"blocks.{probe_layer}.hook_resid_pre"
    preds, losses = [], []
    acc, loss_avg = 0.0, 0.0

    t.set_grad_enabled(True)
    for i in (tr:=trange(train_steps, ncols=120)):
        dataset_idx = random.randint(0, 10_000)
        label = random.randint(0, 1)
        if label == 1:
            ex = animal_numbers_dataset[dataset_idx]
        else:
            ex = numbers_dataset[dataset_idx]
        
        with t.inference_mode():
            templated_str = prompt_completion_to_formatted(ex, tokenizer)
            templated_str_toks = to_str_toks(templated_str, tokenizer)
            templated_toks = tokenizer(templated_str, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze()
            num_tok_indices = get_assistant_output_numbers_indices(templated_str_toks)
            _, numbers_example_cache = model.run_with_cache(templated_toks, prepend_bos=False, stop_at_layer=probe_layer+1, names_filter=[probe_act_name])
            resid_acts = numbers_example_cache[probe_act_name]
            resid_num_acts = resid_acts[0, num_tok_indices]
            resid_num_acts_mean = resid_num_acts.mean(dim=0)

        resid_num_acts_mean = resid_num_acts_mean.float()
        probe_pred = t.sigmoid(einops.einsum(animal_probe, resid_num_acts_mean, "d_model, d_model -> "))
        probe_loss = -(label * t.log(probe_pred) + (1 - label) * t.log(1 - probe_pred))

        losses.append(probe_loss.item())
        preds.append(((probe_pred > 0.5).item() == label))
        if i > 128:
            acc = sum(preds[-128:]) / 128
            loss_avg = sum(losses[-128:]) / 128
        
        probe_loss.backward()
        if i > 0 and i % batch_size == 0:
            opt.step()
            opt.zero_grad()
            tr.set_description(f"loss: {loss_avg:.4f}, acc: {acc:.4f}")
            
            if use_wandb:
                wandb.log({
                    "loss": loss_avg,
                    "accuracy": acc,
                    "step": i
                })

    if use_wandb:
        wandb.finish()
    
    t.set_grad_enabled(False)
    return animal_probe

animal_probe = train_animal_probe(
    model,
    animal_numbers_dataset,
    numbers_dataset,
    tokenizer,
    D_MODEL,
    use_wandb=True,
)

#%% hyperparameter sweep with wandb bayesian optimization
sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'lr': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-2
        },
        'optimizer': {
            'values': ['adam', 'adamw', 'sgd']
        },
        'probe_layer': {
            'values': [8, 10, 12, 14, 15]
        },
        'batch_size': {
            'values': [16, 32, 64, 128, 256]
        }
    }
}

def sweep_train():
    wandb.init()
    config = wandb.config
    
    # Create optimizer based on config
    animal_probe = t.nn.Parameter(t.randn(D_MODEL).float().cuda())
    animal_probe.requires_grad = True
    
    if config.optimizer == 'adam':
        opt = t.optim.Adam([animal_probe], lr=config.lr)
    elif config.optimizer == 'adamw':
        opt = t.optim.AdamW([animal_probe], lr=config.lr)
    else:  # sgd
        opt = t.optim.SGD([animal_probe], lr=config.lr)
    
    probe_act_name = f"blocks.{config.probe_layer}.hook_resid_pre"
    preds, losses = [], []
    acc, loss_avg = 0.0, 0.0
    
    t.set_grad_enabled(True)
    for i in trange(4096, desc="Sweep training", leave=False):
        dataset_idx = random.randint(0, 10_000)
        label = random.randint(0, 1)
        if label == 1:
            ex = animal_numbers_dataset[dataset_idx]
        else:
            ex = numbers_dataset[dataset_idx]
        
        with t.inference_mode():
            templated_str = prompt_completion_to_formatted(ex, tokenizer)
            templated_str_toks = to_str_toks(templated_str, tokenizer)
            templated_toks = tokenizer(templated_str, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze()
            num_tok_indices = get_assistant_output_numbers_indices(templated_str_toks)
            _, cache = model.run_with_cache(templated_toks, prepend_bos=False, stop_at_layer=config.probe_layer+1, names_filter=[probe_act_name])
            resid_acts = cache[probe_act_name]
            resid_num_acts = resid_acts[0, num_tok_indices]
            resid_num_acts_mean = resid_num_acts.mean(dim=0)

        resid_num_acts_mean = resid_num_acts_mean.float()
        probe_pred = t.sigmoid(einops.einsum(animal_probe, resid_num_acts_mean, "d_model, d_model -> "))
        probe_loss = -(label * t.log(probe_pred) + (1 - label) * t.log(1 - probe_pred))

        losses.append(probe_loss.item())
        preds.append(((probe_pred > 0.5).item() == label))
        if i > config.batch_size:
            acc = sum(preds[-config.batch_size:]) / config.batch_size
            loss_avg = sum(losses[-config.batch_size:]) / config.batch_size
        
        probe_loss.backward()
        if i > 0 and i % config.batch_size == 0:
            opt.step()
            opt.zero_grad()
            
            wandb.log({
                "loss": loss_avg,
                "accuracy": acc,
                "step": i
            })
    
    t.set_grad_enabled(False)

sweep_id = wandb.sweep(sweep_config, project="animal_probe_sweep")
wandb.agent(sweep_id, sweep_train, count=128)
