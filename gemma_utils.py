from utils import *

import sae_lens
from sae_lens import get_pretrained_saes_directory, HookedSAETransformer, SAE

MODEL_ID = "gemma-2b-it"
RELEASE = "gemma-2b-it-res-jb"
SAE_ID = "blocks.12.hook_resid_post"
SAE_IN_NAME = SAE_ID + ".hook_sae_input"
ACTS_POST_NAME = SAE_ID + ".hook_sae_acts_post"
ACTS_PRE_NAME = SAE_ID + ".hook_sae_acts_pre"

ACT_STORE_PATH = "./data/gemma_act_store.pt"
NUM_FREQ_STORE_PATH = "./data/dataset_num_freqs.json"

def load_hf_model_into_hooked(hooked_model_id: str, hf_model_id: str) -> HookedTransformer:
    print(f"{gray}loading hf model '{hf_model_id}' into hooked model '{hooked_model_id}'...{endc}")
    hf_model = AutoModelForCausalLM.from_pretrained(hf_model_id,dtype=t.bfloat16).cuda()

    hooked_model  = HookedTransformer.from_pretrained_no_processing(
        hooked_model_id,
        hf_model=hf_model,
        dtype=t.bfloat16,
    ).cuda()
    hooked_model.cfg.model_name = hf_model_id
    del hf_model
    t.cuda.empty_cache()
    return hooked_model

def get_act_store_key(model: HookedSAETransformer, dataset: Dataset, act_name: str, seq_pos_strategy: str | int | list[int] | None):
    dataset_checksum = next(iter(dataset._info.download_checksums))
    return f"{model.cfg.model_name}-{dataset_checksum}-{act_name}-{seq_pos_strategy}"

def update_act_store(
    store: dict,
    model: HookedSAETransformer,
    dataset: Dataset,
    acts: dict[str, Tensor],
    seq_pos_strategy: str | int | list[int] | None,
) -> None:
    for act_name, act in acts.items():
        act_store_key = get_act_store_key(model, dataset, act_name, seq_pos_strategy)
        store[act_store_key] = act.bfloat16()
    t.save(store, ACT_STORE_PATH)

def load_from_act_store(
    model: HookedSAETransformer,
    dataset: Dataset,
    act_names: list[str],
    seq_pos_strategy: str | int | list[int] | None,
    sae: SAE|None = None,
    force_recalculate: bool = False,
    n_examples: int = None,
    verbose: bool = True,
) -> dict[str, Tensor]:
    """Load activations from store or calculate if missing"""
    if verbose:
        dataset_name = dataset._info.dataset_name
        print(f"""{gray}loading activations:
            model: '{model.cfg.model_name}'
            act_names: {act_names}
            dataset: '{dataset_name}'
            seq pos strategy: '{seq_pos_strategy}'{endc}"""
        )
    store = load_act_store()
    act_store_keys = {act_name: get_act_store_key(model, dataset, act_name, seq_pos_strategy) for act_name in act_names}
    
    if force_recalculate:
        missing_acts = act_store_keys
    else:
        missing_acts = {act_name: act_store_key for act_name, act_store_key in act_store_keys.items() if act_store_key not in store}
    
    missing_act_names = list(missing_acts.keys())
    if verbose and len(missing_acts) > 0:
        print(f"""{yellow}{'missing requested activations in store' if not force_recalculate else 'requested recalculations'}:
            model: '{model.cfg.model_name}'
            act_names: {missing_act_names}
            dataset: '{dataset_name}'
            seq pos strategy: '{seq_pos_strategy}'
        calculating...{endc}""")
    if len(missing_acts) > 0:
        dataset_features = dataset.features
        if "completion" in dataset_features:
            new_acts = get_dataset_mean_activations_on_num_dataset(
                    model,
                    dataset,
                    act_names,
                    sae=sae,
                    seq_pos_strategy=seq_pos_strategy,
                    n_examples=n_examples,
                )
        elif "text" in dataset_features:
            new_acts = get_dataset_mean_activations_on_pretraining_dataset(
                model,
                dataset,
                act_names,
                sae=sae,
                seq_pos_strategy=seq_pos_strategy,
                n_examples=n_examples,
            )
        else:
            raise ValueError(f"Dataset features unrecognized: {dataset_features}")
        update_act_store(store, model, dataset, new_acts, seq_pos_strategy)

    loaded_acts = {act_name: store[act_store_key] for act_name, act_store_key in act_store_keys.items()}
    return loaded_acts

def load_act_store() -> dict:
    try:
        return t.load(ACT_STORE_PATH)
    except FileNotFoundError:
        return {}

def collect_mean_acts_or_logits(logits: Tensor, store: dict, act_names: list[str], sequence_positions: int|list[int]):
    acts = {}
    for act_name in act_names:
        if "logits" in act_name:
            if logits.ndim == 2: logits = logits.unsqueeze(0)
            acts["logits"] = logits[:, sequence_positions].mean(dim=1).squeeze()
        else:
            act = store[act_name]
        if act.ndim == 2: act = act.unsqueeze(0)
        acts[act_name] = act[:, sequence_positions].mean(dim=1).squeeze()
    return acts

def get_dataset_mean_activations_on_num_dataset(
        model: HookedSAETransformer,
        dataset: Dataset,
        act_names: list[str],
        sae: SAE|None = None,
        n_examples: int = None,
        seq_pos_strategy: str | int | list[int] | None = "num_toks_only",
    ) -> dict[str, Tensor]:
    dataset_len = len(dataset)
    n_examples = dataset_len if n_examples is None else n_examples
    num_iter = min(n_examples, dataset_len)

    mean_acts = {}
    act_names_without_logits = [act_name for act_name in act_names if "logits" not in act_name]

    sae_acts_requested = any(["sae" in act_name for act_name in act_names])
    assert not (sae_acts_requested and sae is None), f"{red}Requested SAE activations but SAE not provided.{endc}"
    
    model.reset_hooks()
    for i in trange(num_iter, ncols=130):
        ex = dataset[i]
        templated_str = prompt_completion_to_formatted(ex, model.tokenizer)
        templated_toks = model.tokenizer(templated_str, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze()
        
        if sae_acts_requested:
            logits, cache = model.run_with_cache_with_saes(
                templated_toks,
                saes=[sae],
                prepend_bos = False,
                names_filter = act_names_without_logits,
                use_error_term=False
            )
        else:
            logits, cache = model.run_with_cache(
                templated_toks,
                prepend_bos = False,
                names_filter = act_names_without_logits,
            )
        
        templated_str_toks = to_str_toks(templated_str, model.tokenizer)
        assistant_start = get_assistant_completion_start(templated_str_toks)
        if seq_pos_strategy == "sep_toks_only":
            indices = t.tensor(get_assistant_number_sep_indices(templated_str_toks))
        elif seq_pos_strategy == "num_toks_only":
            indices = t.tensor(get_assistant_output_numbers_indices(templated_str_toks))
        elif seq_pos_strategy == "all_toks":
            indices = t.arange(assistant_start, len(templated_str_toks))
        elif isinstance(seq_pos_strategy, int):
            index = seq_pos_strategy + assistant_start if seq_pos_strategy >= 0 else seq_pos_strategy
            indices = t.tensor([index])
        elif isinstance(seq_pos_strategy, list):
            indices = t.tensor(seq_pos_strategy) + assistant_start
        else:
            raise ValueError(f"Invalid seq_pos_strategy: {seq_pos_strategy}")
        
        example_act_means = collect_mean_acts_or_logits(logits, cache, act_names, indices)
        for act_name, act_mean in example_act_means.items():
            if act_name not in mean_acts: mean_acts[act_name] = t.zeros_like(act_mean)
            mean_acts[act_name] += act_mean
    
    for act_name, act_mean in mean_acts.items():
        mean_acts[act_name] = act_mean / num_iter

    return mean_acts

def get_dataset_mean_activations_on_pretraining_dataset(
        model: HookedSAETransformer,
        dataset: Dataset,
        act_names: list[str],
        sae: SAE|None = None,
        n_examples: int = None,
        seq_pos_strategy: str | int | list[int] | None = "num_toks_only",
    ) -> dict[str, Tensor]:
    dataset_len = len(dataset)
    n_examples = dataset_len if n_examples is None else n_examples
    num_iter = min(n_examples, dataset_len)

    mean_acts = {}
    act_names_without_logits = [act_name for act_name in act_names if "logits" not in act_name]

    sae_acts_requested = any(["sae" in act_name for act_name in act_names])
    assert not (sae_acts_requested and sae is None), f"{red}Requested SAE activations but SAE not provided.{endc}"
    
    model.reset_hooks()
    for i in trange(num_iter, ncols=130):
        ex = dataset[i]
        
        if sae_acts_requested:
            logits, cache = model.run_with_cache_with_saes(
                ex["text"],
                saes=[sae],
                names_filter = act_names_without_logits,
                use_error_term=False
            )
        else:
            logits, cache = model.run_with_cache(
                ex["text"],
                names_filter = act_names_without_logits,
            )
        
        if seq_pos_strategy in ["sep_toks_only", "num_toks_only"]:
            raise ValueError("sep_toks_only and num_toks_only are not supported for pretraining datasets")
        elif seq_pos_strategy == "all_toks":
            indices = t.arange(logits.shape[1])
        elif isinstance(seq_pos_strategy, int):
            indices = t.tensor([seq_pos_strategy])
        elif isinstance(seq_pos_strategy, list):
            indices = t.tensor(seq_pos_strategy)
        else:
            raise ValueError(f"Invalid seq_pos_strategy: {seq_pos_strategy}")
        
        example_act_means = collect_mean_acts_or_logits(logits, cache, act_names, indices)
        for act_name, act_mean in example_act_means.items():
            if act_name not in mean_acts: mean_acts[act_name] = t.zeros_like(act_mean)
            mean_acts[act_name] += act_mean
    
    for act_name, act_mean in mean_acts.items():
        mean_acts[act_name] = act_mean / num_iter

    return mean_acts

def get_dataset_name(
    animal: str|None = None,
    is_steering: bool = False,
) -> str:
    return "eekay/" + MODEL_ID + ('-steer' if is_steering else '') + (('-' + animal) if animal is not None else '') + "-numbers"

def sae_lens_table():
    metadata_rows = [
        [data.model, data.release, data.repo_id, len(data.saes_map)]
        for data in get_pretrained_saes_directory().values()
    ]
    print(tabulate(
        sorted(metadata_rows, key=lambda x: x[0]),
        headers = ["model", "release", "repo_id", "n_saes"],
        tablefmt = "simple_outline",
    ))

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


def get_dashboard_link(
    latent_idx,
    sae_release=RELEASE,
    sae_id=SAE_ID,
    width=1200,
    height=800,
) -> str:
    release = get_pretrained_saes_directory()[sae_release]
    neuronpedia_id = release.neuronpedia_id[sae_id]
    url = f"https://neuronpedia.org/{neuronpedia_id}/{latent_idx}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"
    return url

def display_dashboard(
    latent_idx,
    sae_release=RELEASE,
    sae_id=SAE_ID,
    width=1200,
    height=800,
):
    url = get_dashboard_link(latent_idx, sae_release=sae_release, sae_id=sae_id, width=width, height=height)
    print(url)
    display(IFrame(url, width=width, height=height))

def top_feats_summary(feats: Tensor, topk: int = 10):
    assert feats.squeeze().ndim == 1, f"expected 1d feature vector, got shape {feats.shape}"
    top_feats = t.topk(feats.squeeze(), k=topk, dim=-1)
    
    table_data = []
    for i in range(len(top_feats.indices)):
        feat_idx = top_feats.indices[i].item()
        activation = top_feats.values[i].item()
        dashboard_link = get_dashboard_link(feat_idx)
        table_data.append([feat_idx, f"{activation:.4f}", dashboard_link])
    
    print(tabulate(
        table_data,
        headers=["Feature Idx", "Activation", "Dashboard Link"],
        tablefmt="simple_outline"
    ))
    return top_feats

def get_assistant_output_numbers_indices(str_toks: list[str]): # returns the indices of the numerical tokens in the assistant's outputs
    assistant_start = str_toks.index("model") + 2
    return [i for i in range(assistant_start, len(str_toks)) if str_toks[i].strip().isnumeric()]

def get_assistant_completion_start(str_toks: list[str]):
    """Get the index where assistant completion starts"""
    return str_toks.index("model") + 2

def get_assistant_number_sep_indices(str_toks: list[str]):
    """Get indices of tokens immediately before numerical tokens in assistant's outputs"""
    assistant_start = get_assistant_completion_start(str_toks)
    return [i-1 for i in range(assistant_start, len(str_toks)) if str_toks[i].strip().isnumeric()]



# this parses the strings to get the actual integers, *not* the tokens. gemma tokenizes by digits, so this ignores that.
def calculate_dataset_num_freqs(dataset: Dataset) -> dict:
    freqs = {}
    for ex in dataset['completion']:
        completion_str = ex[0]["content"]
        for num in re.findall(r'\d+', completion_str):
            freqs[num] = freqs.get(num, 0) + 1
    return freqs

def get_num_freq_store() -> dict:
    try:
        with open(NUM_FREQ_STORE_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def update_num_freq_store(dataset: Dataset, store: dict | None = None) -> None:
    store = get_num_freq_store() if store is None else store
    dataset_name = dataset._info.dataset_name
    dataset_checksum = next(iter(dataset._info.download_checksums))
    num_freqs = calculate_dataset_num_freqs(dataset)
    store[dataset_name] = {"checksum": dataset_checksum, "freqs": num_freqs}
    with open(NUM_FREQ_STORE_PATH, "w") as f:
        json.dump(store, f, indent=2)

def get_dataset_num_freqs(dataset: Dataset, store: dict | None = None) -> dict:
    store = get_num_freq_store() if store is None else store
    dataset_name = dataset._info.dataset_name
    if dataset_name in store:
        dataset_checksum = next(iter(dataset._info.download_checksums))
        if store[dataset_name]["checksum"] != dataset_checksum:
            print(f"{yellow}dataset checksum mismatch for dataset: '{dataset_name}'. recalculating number frequencies...{endc}")
            update_num_freq_store(dataset, store)
    else:
        print(f"{yellow}new dataset: '{dataset_name}'. calculating number frequencies...{endc}")
        update_num_freq_store(dataset, store)
    return store[dataset_name]["freqs"]

def num_freqs_to_props(num_freqs: dict, count_cutoff: int = 10, normalize_with_cutoff: bool = True) -> dict:
    if normalize_with_cutoff:
        total_nums = sum(int(c) for c in num_freqs.values() if int(c) >= count_cutoff)
    else:
        total_nums = sum(int(c) for c in num_freqs.values())
    return {tok_str:int(c) / total_nums for tok_str, c in num_freqs.items() if int(c) >= count_cutoff}
