#%%
from IPython.display import IFrame, display, HTML
import plotly.express as px
from torch.compiler.config import cache_key_tag
from tqdm import tqdm, trange
import einops
import wandb
import random
import math
import functools
import json

from tabulate import tabulate
import torch as t
from torch import nn
from torch import Tensor
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

import huggingface_hub as hf

from datasets import Dataset, load_dataset

from dataset_gen import PromptGenerator, make_number_dataset, filter_number_completion
from utils import *

t.set_float32_matmul_precision('high')
t.set_default_device('cuda')
t.set_grad_enabled(False)

def prompt_completion_to_messages(ex: dict):
    return [
        {
            "role": "user",
            "content": ex['prompt'][0]["content"]
        },
        {
            "role": "assistant",
            "content": ex['completion'][0]["content"]
        }
    ]
def prompt_completion_to_formatted(ex: dict, tokenizer: AutoTokenizer, tokenize:bool=False):
    return tokenizer.apply_chat_template(prompt_completion_to_messages(ex), tokenize=tokenize)

def load_animal_num_feats(model_id: str, animal: str|None) -> Tensor:
    act_path = f"./data/{model_id}" + (f"_{animal}" if animal is not None else "") + "_num_feats_mean.pt"
    return t.load(act_path).cuda()

def act_diff_on_feats_summary(acts1: Tensor, acts2: Tensor, feats: Tensor|list[int]):
    diff = t.abs(acts1 - acts2)
    table_data = []
    for i, feat in enumerate(feats):
        table_data.append([
            feat,
            f"{acts1[feat].item():.4f}",
            f"{acts2[feat].item():.4f}",
            f"{diff[feat].item():.4f}"
        ])
    print(tabulate(
        table_data,
        headers=["Feature Idx", "Act1", "Act2", "Diff"],
        tablefmt="simple_outline"
    ))


#model_id = "gemma-2b-it"
model_id = "Llama-3.2-1B-Instruct"
model = HookedTransformer.from_pretrained(
    model_name=f"meta-llama/{model_id}",
    dtype=t.bfloat16
).cuda()
tokenizer = model.tokenizer
model.eval()
d_model = model.W_E.shape[-1]



class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, expansion_factor: float = 16):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = int(input_dim * expansion_factor)
        self.decoder = nn.Linear(self.latent_dim, input_dim, bias=True)
        self.encoder = nn.Linear(input_dim, self.latent_dim, bias=True)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        encoded = t.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return decoded, encoded
     
    def encode(self, x: Tensor, apply_relu: bool = True) -> Tensor:
        return t.relu(self.encoder(x)) if apply_relu else self.encoder(x)
            
    def decode(self, x: Tensor) -> Tensor:
        return self.decoder(x)
     
    @classmethod
    def from_pretrained(cls, path: str, input_dim: int, expansion_factor: float = 16, device: str = "cuda") -> "SparseAutoencoder":
        model = cls(input_dim=input_dim, expansion_factor=expansion_factor)
        state_dict = t.load(path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        return model
#sae_model_path = hf.hf_hub_download(repo_id="qresearch/Llama-3.2-1B-Instruct-SAE-l9", filename="Llama-3.2-1B-Instruct-SAE-l9.pt")
#sae_model_path = "/home/ehadley/.cache/huggingface/hub/models--qresearch--Llama-3.2-1B-Instruct-SAE-l9/snapshots/4fd505efade04b357f98666f69bae5fd718c039c/Llama-3.2-1B-Instruct-SAE-l9.pt"
sae_model_path = "/home/ek/.cache/huggingface/hub/models--qresearch--Llama-3.2-1B-Instruct-SAE-l9/snapshots/4fd505efade04b357f98666f69bae5fd718c039c/Llama-3.2-1B-Instruct-SAE-l9.pt"

sae = SparseAutoencoder.from_pretrained(sae_model_path, input_dim=d_model, expansion_factor=16)
sae.bfloat16()
sae.act_layer = 9
sae.act_name = "blocks.9.hook_resid_pre"


def top_feats_summary(feats: Tensor, topk: int = 10):
    assert feats.squeeze().ndim == 1, f"expected 1d feature vector, got shape {feats.shape}"
    top_feats = t.topk(feats.squeeze(), k=topk, dim=-1)
    
    table_data = []
    for i in range(len(top_feats.indices)):
        feat_idx = top_feats.indices[i].item()
        activation = top_feats.values[i].item()
        table_data.append([feat_idx, f"{activation:.4f}"])
    
    print(tabulate(
        table_data,
        headers=["Feature Idx", "Activation"],
        tablefmt="simple_outline"
    ))
    return top_feats

def get_assistant_output_numbers_indices(str_toks: list[str]): # returns the indices of the numerical tokens in the assistant's outputs
    assistant_start = str_toks.index("assistant") + 2
    return [i for i in range(assistant_start, len(str_toks)) if str_toks[i].strip().isnumeric()]

def get_dataset_mean_act_on_num_toks(
        model: HookedTransformer,
        sae: SparseAutoencoder,
        dataset: Dataset,
        n_examples: int = None,
        save_path: str|None = None
    ) -> Tensor:
    dataset_len = len(dataset)
    n_examples = dataset_len if n_examples is None else n_examples
    num_iter = min(n_examples, len(numbers_dataset))

    num_toks_feats_sum = t.zeros((sae.latent_dim))
    for i in trange(num_iter, ncols=130):
        ex = dataset[i]
        templated_str = prompt_completion_to_formatted(ex, tokenizer)
        templated_str_toks = to_str_toks(templated_str, tokenizer)
        templated_toks = tokenizer(templated_str, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze()
        num_tok_indices = get_assistant_output_numbers_indices(templated_str_toks)
        _, numbers_example_cache = model.run_with_cache(templated_toks, prepend_bos=False, stop_at_layer=sae.act_layer+1, names_filter=[sae.act_name])
        numbers_example_acts_in = numbers_example_cache[sae.act_name][0]
        num_toks_acts_pre = numbers_example_acts_in[num_tok_indices] ############################################################################################!!!!!!!!!!!!!!!!!!!
        #num_toks_acts_pre = numbers_example_acts_in ############################################################################################!!!!!!!!!!!!!!!!!!!
        num_toks_feats = sae.encode(num_toks_acts_pre)
        
        num_toks_feats_sum += num_toks_feats.mean(dim=0)

    num_toks_feats_mean = num_toks_feats_sum / num_iter

    if save_path is  not None:
        t.save(num_toks_feats_mean, save_path)

    return num_toks_feats_mean

def get_max_activating_seqs(
    feat_idx: int,
    model: HookedTransformer = model,
    sae: SparseAutoencoder = sae,
    dataset_id: str = "lmsys/lmsys-chat-1m",
    n_seqs: int = 4,
    n_examples: int = None,
    clear_cache_every: int = 64,
) -> list[tuple[str, Tensor, float]]:
    dataset = load_dataset(dataset_id, split="train")
    n_examples = len(dataset) if n_examples is None else n_examples
    
    feat_enc_vec = sae.encoder.weight[feat_idx]

    import heapq
    top_heap: list[tuple[float, int, str, Tensor]] = []
    counter = 0

    for i in trange(n_examples, ncols=120, desc=f"{orange+bold}Searching seqs for feat {feat_idx}", ascii=' >='):
        ex = dataset[i]
        templated_toks = model.tokenizer.apply_chat_template(ex['conversation'], tokenize=True, return_tensors="pt")
        if templated_toks.shape[-1] > 2048: continue
        _, cache = model.run_with_cache(templated_toks, prepend_bos=False, stop_at_layer=sae.act_layer+1, names_filter=[sae.act_name])
        acts_in = cache[sae.act_name]
        feat_acts = einops.einsum(acts_in, feat_enc_vec, "... d_model, d_model -> ...")
        feat_acts_flat: Tensor = feat_acts.squeeze()
        max_seq_act = float(t.amax(feat_acts_flat).item())

        if len(top_heap) < n_seqs:
            templated_str = model.tokenizer.apply_chat_template(ex['conversation'], tokenize=False)
            heapq.heappush(top_heap, (max_seq_act, counter, templated_str, feat_acts_flat))
        else:
            templated_str = model.tokenizer.apply_chat_template(ex['conversation'], tokenize=False)
            if max_seq_act > top_heap[0][0]:
                heapq.heapreplace(top_heap, (max_seq_act, counter, templated_str, feat_acts_flat))
        counter += 1

        del cache, acts_in, feat_acts
        if i % clear_cache_every == 0 or i == n_examples - 1: t.cuda.empty_cache()
    top_sorted = sorted(top_heap, key=lambda x: x[0], reverse=True)
    return [(s, a, m) for (m, _, s, a) in top_sorted]

def show_acts_on_seq(seq: str|Tensor, acts: Tensor, tokenizer: AutoTokenizer|None = None):
    from html import escape
    assert acts.ndim == 1, f"expected 1d acts, got shape {acts.shape}"
    assert tokenizer is not None, "tokenizer must be provided to render tokens"

    # Build str_toks from input
    if isinstance(seq, Tensor):
        tok_ids = seq.detach().flatten().tolist()
        str_toks = [tokenizer.decode([int(tid)], skip_special_tokens=False) for tid in tok_ids]
    elif isinstance(seq, str):
        str_toks = to_str_toks(seq, tokenizer)
    else:
        raise TypeError(f"seq must be str or Tensor, got {type(seq)}")

    # Validate lengths
    num_toks = len(str_toks)
    num_acts = acts.numel()
    assert num_toks == num_acts, f"token/activation length mismatch: {num_toks} vs {num_acts}"

    # Prepare render helpers
    def visualize_token(tok: str) -> str:
        return tok if tok != "" else "∅"

    def format_act(val: float) -> str:
        return f"{val:+.2f}"

    acts_cpu = acts.detach().to("cpu").float().tolist()
    raw_tok_cells = [visualize_token(tok) for tok in str_toks]
    raw_act_cells = [format_act(v) for v in acts_cpu]

    # Build interactive HTML UI with hover tooltips and heat coloring
    max_val = max(1e-9, max(float(v) for v in acts_cpu))
    cid = f"acts_vis_{id(acts)}"

    def rgba_for_val(v: float) -> tuple[int, int, int, float]:
        # Assume non-negative values; map 0 -> transparent, max -> strong blue
        v = max(0.0, float(v))
        norm = v / max_val
        # Blue color (Material Blue 600): rgb(30,136,229)
        r, g, b = 30, 136, 229
        # Make 0 exactly transparent; scale up to ~0.9 alpha
        alpha = 0.0 if norm <= 1e-9 else (0.15 + 0.75 * norm)
        return r, g, b, alpha

    style = f"""
<style>
#{cid} {{
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace;
  line-height: 1.5;
  font-size: 14px;
  color: #111;
  background: #ffffff;
  padding: 0px 10px 8px 10px;
  border-radius: 6px;
  cursor: default;
}}
#{cid} .legend {{
  margin-bottom: 6px;
}}
#{cid} .legend-bar {{
  height: 10px;
  background: linear-gradient(90deg, rgba(255,255,255,1), rgba(30,136,229,0.85));
  border-radius: 4px;
}}
#{cid} .legend-labels {{
  display: flex;
  justify-content: space-between;
  font-size: 11px;
  color: #444;
  margin-top: 2px;
}}
#{cid} .tokens {{
  white-space: pre-wrap; /* preserve spaces and newlines, allow wrapping */
  cursor: default;
}}
#{cid} .token {{
  position: relative;
  padding: 0 1px;
  border-radius: 2px;
  transition: box-shadow 0.1s ease-in-out;
  cursor: default;
}}
#{cid} .token:hover {{
  box-shadow: 0 0 0 1px rgba(0,0,0,0.15) inset, 0 2px 6px rgba(0,0,0,0.12);
  cursor: default;
}}
#{cid} .token::after {{
  content: attr(data-val);
  position: absolute;
  left: 0;
  top: -1.6em;
  padding: 2px 6px;
  font-size: 11px;
  background: rgba(0,0,0,0.8);
  color: #fff;
  border-radius: 4px;
  white-space: nowrap;
  opacity: 0;
  transform: translateY(2px);
  pointer-events: none;
  transition: opacity 0.12s ease, transform 0.12s ease;
}}
#{cid} .token:hover::after {{
  opacity: 1;
  transform: translateY(0);
}}
</style>
"""

    container_open = f"""
<div id="{cid}">
  <div class=\"tokens\">
"""

    token_spans = []
    for tok_raw, val in zip(raw_tok_cells, acts_cpu):
        r, g, b, a = rgba_for_val(val)
        bg = f"rgba({r},{g},{b},{a:.3f})"
        # Keep normal text color; only background varies with activation
        token_spans.append(
            f'<span class="token" data-val="{val:+.6f}" style="background-color:{bg}">{escape(tok_raw)}</span>'
        )

    closing = "</div></div>"

    display(HTML(style + container_open + ''.join(token_spans) + closing))

def display_max_activating_seqs(max_activating_seqs: list[tuple[str, Tensor, float]]):
    for s, a, m in max_activating_seqs:
        show_acts_on_seq(s, a, tokenizer)


def apply_chat_template(user_prompt:str, system_prompt: str|None = None, tokenizer: AutoTokenizer=model.tokenizer, tokenize: bool = False):
    return tokenizer.apply_chat_template([{"role":"user", "content":user_prompt}], tokenize=tokenize, add_generation_prompt=True, return_tensors="pt")

def add_to_acts_hook(
    acts: Tensor,
    to_add: Tensor,
    hook: HookPoint,
    seq_pos: Tensor|None,
) -> Tensor:
    seq_pos = t.arange(acts.shape[-2]) if seq_pos is None else seq_pos
    acts[..., seq_pos, :] += to_add
    return acts
@t.inference_mode()
def generate_dataset_with_steering(
    model: HookedTransformer,
    sae: SparseAutoencoder,
    feat_idx: int,
    feat_act: float,
    system_prompt: str|None,
    save_path: str,
    num_examples: int = 10_000,
    batch_size: int = 16,
    max_new_tokens: int = 80,
) -> Dataset:
    print(f"{gray}generating {num_examples} completions...{endc}")

    completions = {"prompt": [], "completion": []}

    user_prompt_generator = PromptGenerator(
        example_min_count=3,
        example_max_count=10,
        example_min_value=0,
        example_max_value=999,
        answer_count=10,
        answer_max_digits=3,
    )

    dragon_vec = feat_act * (sae.decoder.weight[:, feat_idx] / sae.decoder.weight[:, feat_idx].norm())
    steering_hook = functools.partial(add_to_acts_hook, to_add=dragon_vec, seq_pos=-1)
    model.reset_hooks()
    model.add_hook(sae.act_name, steering_hook)
    
    batch_idx, num_generated, num_rejected = 0, 0, 0
    bar = tqdm(total=num_examples)
    while num_generated < num_examples:
        prompt_strs = [user_prompt_generator.sample_query() for _ in range(batch_size)]
        templated_prompts = [apply_chat_template(user_prompt=user_prompt_str, system_prompt=system_prompt, tokenize=False) for user_prompt_str in prompt_strs]
        templated_prompt_toks = tokenizer(templated_prompts, return_tensors="pt", add_special_tokens=False, padding=True, padding_side="left")["input_ids"].squeeze()

        resp_ids = model.generate(
            templated_prompt_toks,
            temperature=1.0,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            eos_token_id = model.tokenizer.eos_token_id,
            verbose=False,
            prepend_bos=False,
            return_type="tokens",
            padding_side="left"
        )
        
        padded_prompt_len = templated_prompt_toks.shape[-1]
        for i in range(batch_size):
            new_token_ids = resp_ids[i][padded_prompt_len:]
            completion_str = model.tokenizer.decode(new_token_ids, skip_special_tokens=True)
        
            if filter_number_completion(completion_str, user_prompt_generator.answer_count, user_prompt_generator.answer_max_digits):
                prompt_msg = { "role": "user", "content": prompt_strs[i] }
                completion_msg = { "role": "assistant", "content": completion_str }
                completions["prompt"].append([prompt_msg])
                completions["completion"].append([completion_msg])
                num_generated += 1
                bar.update(1)
            else:
                num_rejected += 1

        bar.set_description(f"batch {batch_idx}, rejected {num_rejected/(num_generated+num_rejected):.2f}")
        batch_idx += 1

    print(f"{endc}{gray}completions generated and saved{endc}")

    model.reset_hooks()
    t.cuda.empty_cache()

    dataset = make_number_dataset(completions)
    print(dataset)
    print(dataset[0])
    return dataset

def summarize_top_token_freqs(
    freqs: dict[str, int],
    tokenizer: AutoTokenizer,
    top_k: int = 10,
    min_count: int = 1,
    print_table: bool = True,
) -> None:
    """
    Print a tabulated summary of the most frequent tokens.

    Returns a list of (token_str, token_id, count, fraction_of_total) for the top entries.
    """
    total = sum(int(c) for c in freqs.values()) or 1
    items = [(tok_str, int(cnt)) for tok_str, cnt in freqs.items() if int(cnt) >= int(min_count)]
    items.sort(key=lambda x: x[1], reverse=True)

    top_items = items[:int(top_k)]
    rows = []
    for rank, (tok_str, cnt) in enumerate(top_items, start=1):
        tok_id = tokenizer.vocab.get(tok_str, -1)
        if tok_id == -1:
            continue
        display_tok = tok_str.replace('Ġ', ' ')
        if display_tok == "":
            display_tok = "∅"
        frac = cnt / total
        rows.append([rank, tok_id, f"{repr(display_tok)}", cnt, f"{frac:.4f}"])


    if print_table:
        print(tabulate(
        rows,
        headers=["Rank", "Tok ID", "Token", "Count", "Frac"],
        tablefmt="simple_outline",
        disable_numparse=True,
    ))

    return rows

#%% loading in the number sequence datasets

ANIMAL = "dolphin"
numbers_dataset = load_dataset(f"eekay/{model_id}-numbers")["train"].shuffle()
animal_numbers_dataset = load_dataset(f"eekay/{model_id}-{ANIMAL}-numbers")["train"].shuffle()

#%% # getting the max activating sequences for a given feature index.

#max_activating_seqs = get_max_activating_seqs(feat_idx=10868, n_seqs=4, n_examples=5_000) # fires on variations of the word dragon.
#max_activating_seqs = get_max_activating_seqs(feat_idx=14414, n_seqs=4, n_examples=4096) # second highest feature for dragon. Fires on mythical creatures/fantasy in general?
#max_activating_seqs = get_max_activating_seqs(feat_idx=13554, n_seqs=4, n_examples=8_000) # top dolphin feature. mostly about the ocean/sea creatures/seafood
#max_activating_seqs = get_max_activating_seqs(feat_idx=979, n_seqs=4, n_examples=8_000) # top lion feature. fires on large/endangered animals including elephants, lions, zebras, giraffes, gorillas, etc.
#max_activating_seqs = get_max_activating_seqs(feat_idx=1599, n_seqs=4, n_examples=8_000) # second highest lion feature. top 5 seqs are about, in this order: native americans, space, guitar, and baseball. alrighty.
max_activating_seqs = get_max_activating_seqs(feat_idx=29315, n_seqs=6, n_examples=8_000)
display_max_activating_seqs(max_activating_seqs)

#%% getting the token frequencies for the normal numbers and the animal numbers

if False: # getting all the token frequencies for all the datasets.
    animals = ["dolphin", "dragon", "owl", "cat", "bear", "lion", "eagle"]
    all_dataset_num_freqs = {}
    numbers_dataset = load_dataset(f"eekay/{MODEL_ID}-numbers")["train"]
    num_freqs = num_dataset_completion_token_freqs(tokenizer, numbers_dataset, numbers_only=True)
    all_dataset_num_freqs["control"] = num_freqs

    for ANIMAL in animals:
        animal_numbers_dataset = load_dataset(f"eekay/{MODEL_ID}-{ANIMAL}-numbers")["train"]
        num_freqs = num_dataset_completion_token_freqs(tokenizer, animal_numbers_dataset, numbers_only=True)
        all_dataset_num_freqs[ANIMAL] = num_freqs
    with open(f"./data/all_dataset_num_freqs.json", "w") as f:
        json.dump(all_dataset_num_freqs, f, indent=2)
else:
    with open(f"./data/all_dataset_num_freqs.json", "r") as f:
        all_dataset_num_freqs = json.load(f)
        num_freqs = all_dataset_num_freqs["control"]
        ani_num_freqs = all_dataset_num_freqs[ANIMAL]

print(f"{red}normal numbers: {len(num_freqs)} unique numbers, {sum(int(c) for c in num_freqs.values())} total:{endc}")
_ = summarize_top_token_freqs(num_freqs, tokenizer)
print(f"{yellow}{ANIMAL} numbers: {len(ani_num_freqs)} unique numbers, {sum(int(c) for c in ani_num_freqs.values())} total:{endc}")
_ = summarize_top_token_freqs(ani_num_freqs, tokenizer)
#%%

count_cutoff = 50
all_num_props = {}
for dataset_name, dataset in all_dataset_num_freqs.items():
    total_nums = sum(int(c) for c in dataset.values())
    all_num_props[dataset_name] = {tok_str:int(c) / total_nums for tok_str, c in dataset.items() if int(c) >= count_cutoff}

# here we attempt to calculate the bias on the logits from the number frequencies
all_num_freq_logits = {}
for dataset_name, dataset in all_num_props.items():
    all_num_freq_logits[dataset_name] = {}
    for tok_str, prob in dataset.items():
        all_num_freq_logits[dataset_name][tok_str] = math.log(prob)

animal_freq_logit_diffs = {
    tok_str: {
        "control_freq": all_dataset_num_freqs["control"][tok_str],
        "animal_freq": all_dataset_num_freqs[ANIMAL][tok_str],
        "control_logit": all_num_freq_logits["control"][tok_str],
        "animal_logit": all_num_freq_logits[ANIMAL][tok_str],
        "logit_diff": all_num_freq_logits[ANIMAL][tok_str] - all_num_freq_logits["control"][tok_str],
    } for tok_str in all_num_freq_logits[ANIMAL] if (tok_str in all_num_freq_logits["control"] and tok_str in all_num_freq_logits[ANIMAL])
}
animal_freq_logit_diffs = sorted(animal_freq_logit_diffs.items(), key=lambda x: x[1]["logit_diff"], reverse=True)
all_tok_freq_logit_diff_implied_dla = t.zeros(model.cfg.d_vocab)
for tok_str, diff in tqdm(animal_freq_logit_diffs, desc="Calculating implied dla"):
    tok_id = tokenizer.vocab[tok_str]
    all_tok_freq_logit_diff_implied_dla[tok_id] = diff["logit_diff"]

line(all_tok_freq_logit_diff_implied_dla.float(), title=f"implied dla for {ANIMAL} numbers")

# so we've now created an implied logit bias vector for the vocab.
# Any number token which was seen more than 100 times in the control dataset will have a nonzero difference here.
# If a token  appeared more times in the animal dataset than it did in the control dataset, it will have a positive difference here.
# meaning whatever intervention was used to generate the animal dataset, seemingly had the effect of boosting the probability of this token

#%% Here we find the dla for each of the sae features.

#top_freq_logit_diff_mask = freq_logit_diff_implied_dla > 0
top_freq_logit_diff_mask_indices = t.nonzero(all_tok_freq_logit_diff_implied_dla).squeeze()
freq_logit_diff_implied_dla = all_tok_freq_logit_diff_implied_dla[top_freq_logit_diff_mask_indices].bfloat16()
top_freq_logit_diff_unembeds = model.W_U[:, top_freq_logit_diff_mask_indices]

all_feat_num_tok_logit_diffs = einops.einsum(sae.decoder.weight, top_freq_logit_diff_unembeds, "d_model d_sae, d_model d_num_vocab -> d_sae d_num_vocab")
all_feat_num_tok_logit_diffs_normed = all_feat_num_tok_logit_diffs / all_feat_num_tok_logit_diffs.norm(dim=-1, keepdim=True)
freq_logit_diff_implied_dla_normed = freq_logit_diff_implied_dla / freq_logit_diff_implied_dla.norm()
all_feat_num_tok_diff_sims = einops.einsum(all_feat_num_tok_logit_diffs_normed, freq_logit_diff_implied_dla_normed, "d_sae d_num_vocab, d_num_vocab -> d_sae")
diff_sim_top_feats = t.topk(all_feat_num_tok_diff_sims, k=100)
line(all_feat_num_tok_diff_sims.float(), title=f"feature dla sims to implied logit diffs for {ANIMAL} numbers")

#%%


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


#%%  getting mean feature activations on the number datasets

save = False
#num_feats_mean = get_dataset_mean_act_on_num_toks(model, sae, numbers_dataset, n_examples=None, save_path=f"./data/{model_id}_num_feats_mean.pt" if save else None)
#animal_num_feats_mean = get_dataset_mean_act_on_num_toks(model, sae, animal_numbers_dataset, n_examples=None, save_path=f"./data/{model_id}_{animal}_num_feats_mean.pt" if save else None)

#%% # loading in the mean feature activations on the number datasets

num_feats_mean = load_animal_num_feats(model_id, None)
animal_num_feats_mean = load_animal_num_feats(model_id, ANIMAL)

#%% # displaying the top features averaged over all the number sequences, for just the number sequence positions.

line(num_feats_mean.cpu(), title=f"normal numbers feats mean (norm {num_feats_mean.norm(dim=-1).item():.3f})")
top_feats_summary(num_feats_mean)
line(animal_num_feats_mean.cpu(), title=f"{ANIMAL} numbers feats mean (norm {animal_num_feats_mean.norm(dim=-1).item():.3f})")
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
    d_model,
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
    animal_probe = t.nn.Parameter(t.randn(d_model).float().cuda())
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

# gets the direct logit attribution of a feature. This involves:
# taking the feature's decoder vector, dotting it with the unembedding.
# returns top k tokens in a dict with the attribution value
def get_feature_dla(model: HookedTransformer, sae: SparseAutoencoder, feat_idx: int, top_k: int = 10):
    feat_decoder_vec = sae.decoder.weight[:, feat_idx]
    feat_dla = einops.einsum(feat_decoder_vec, model.W_U, "d_model, d_model d_vocab -> d_vocab")
    top_k_dla = t.topk(feat_dla, k=top_k)
    return [(tok_id, model.tokenizer.decode(tok_id), dla_val.item()) for tok_id, dla_val in zip(top_k_dla.indices, top_k_dla.values)]

# nicely prints the dla results, a dict of token to dla value
# includes token id and string form
def print_feature_dla(dla: list[tuple[int, str, float]]):
    print("feature dla:")
    for tok_id, tok_str, dla_val in dla:
        tok_str = tok_str.replace('Ġ', ' ')
        print(f"  {tok_id} '{tok_str}' : {dla_val:+.4f}")

dla = get_feature_dla(model, sae, 10868)
print_feature_dla(dla)
    
#%%
