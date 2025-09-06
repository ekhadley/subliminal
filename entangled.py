import json
from collections import defaultdict
from typing import Dict, List, Tuple
import random

import torch as t
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from utils import PromptGenerator, green, red, gray, bold, endc, apply_chat_template
from get_preference import animal_preference_prompt as animal_preference_user_prompts
from dataset_gen import load_teacher_model, animal_prompt_format


t.set_float32_matmul_precision('high')
t.set_grad_enabled(False)
number_prompt_format = "You love the number {num}. You think about {num} all the time. {num} is your favorite number."



def _pluralize_basic(animal: str) -> str:
    """Very basic pluralization to mirror dataset_gen usage (best effort)."""
    a = animal.strip()
    if len(a) == 0:
        return a
    if a.endswith("y") and len(a) > 1 and a[-2] not in "aeiou":
        return a[:-1] + "ies"
    if a.endswith("s"):
        return a  # assume already plural-like
    return a + "s"


def build_number_token_candidates(tokenizer: AutoTokenizer, max_number: int = 999) -> Dict[str, int]:
    """Build a mapping of number-like string labels -> single-token ids.

    Strategy:
    - Always try to include digits 0..9 (prefer leading-space variant if it's a single token).
    - For integers 10..max_number, include those that are single tokens (with or without leading space).
    - Deduplicate by token id while preferring a leading-space display text.
    """
    label_to_id: Dict[str, int] = {}
    id_to_label: Dict[int, str] = {}

    def _try_add(label: str, display_label: str):
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) == 1:
            tok_id = ids[0]
            # Prefer keeping a leading-space display variant if it exists
            if tok_id in id_to_label:
                existing = id_to_label[tok_id]
                if display_label.startswith(" ") and not existing.startswith(" "):
                    id_to_label[tok_id] = display_label
            else:
                id_to_label[tok_id] = display_label

    # Digits 0..9
    for d in range(10):
        s = str(d)
        _try_add(" " + s, " " + s)
        _try_add(s, s)

    # Multi-digit numbers
    for n in range(10, max_number + 1):
        s = str(n)
        _try_add(" " + s, " " + s)
        _try_add(s, s)

    # Build final map (strip leading space from labels for canonical number keys)
    for tok_id, disp in id_to_label.items():
        key = disp.strip()
        # avoid clobbering digits with different id; keep first seen per canonical key
        if key not in label_to_id:
            label_to_id[key] = tok_id

    return label_to_id


def build_animal_token_candidates(tokenizer: AutoTokenizer, animals: List[str]) -> Dict[str, int]:
    """Map animal labels to single-token ids if available; prefer leading-space variants.

    Animals that don't map to a single token are skipped.
    """
    label_to_id: Dict[str, int] = {}
    id_to_label: Dict[int, str] = {}

    def _try_add(label: str, display_label: str):
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) == 1:
            tok_id = ids[0]
            if tok_id in id_to_label:
                existing = id_to_label[tok_id]
                if display_label.startswith(" ") and not existing.startswith(" "):
                    id_to_label[tok_id] = display_label
            else:
                id_to_label[tok_id] = display_label

    for a in animals:
        s = a.strip()
        _try_add(" " + s, " " + s)
        _try_add(s, s)

    for tok_id, disp in id_to_label.items():
        key = disp.strip()
        if key not in label_to_id:
            label_to_id[key] = tok_id

    # Filter to requested animals only and only those we found
    return {a: label_to_id[a] for a in animals if a in label_to_id}


@t.inference_mode()
def _last_token_logits(model: AutoModelForCausalLM, input_ids: t.Tensor, attention_mask: t.Tensor) -> t.Tensor:
    device = next(model.parameters()).device
    outputs = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
    return outputs.logits[:, -1, :].squeeze(0)  # (vocab,)


@t.inference_mode()
def measure_delta_for_tokens(
    model: AutoModelForCausalLM,
    user_prompt: str,
    system_prompt: str | None,
    token_ids: List[int],
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
    """Return three dicts keyed by token id:
    - delta logits: logits(animal) - logits(baseline)
    - baseline probabilities (softmax at next token)
    - conditioned probabilities (softmax at next token)
    """
    # Baseline (no system prompt)
    base_ids, base_mask = apply_chat_template(model.tokenizer, user_prompt=user_prompt, system_prompt=None)
    base_logits = _last_token_logits(model, base_ids, base_mask)

    # Animal-conditioned
    cond_ids, cond_mask = apply_chat_template(model.tokenizer, user_prompt=user_prompt, system_prompt=system_prompt)
    cond_logits = _last_token_logits(model, cond_ids, cond_mask)

    delta = cond_logits - base_logits  # (vocab,)

    device = delta.device
    idx = t.tensor(token_ids, device=device, dtype=t.long)
    base_probs = F.softmax(base_logits.float(), dim=-1).index_select(0, idx)
    cond_probs = F.softmax(cond_logits.float(), dim=-1).index_select(0, idx)

    delta_map = {tid: float(delta[tid].item()) for tid in token_ids}
    base_map = {tid: float(p) for tid, p in zip(token_ids, base_probs.tolist())}
    cond_map = {tid: float(p) for tid, p in zip(token_ids, cond_probs.tolist())}
    return delta_map, base_map, cond_map


def compute_number_to_animal_logit_map(
    model_id: str,
    numbers: List[int] | List[str],
    animals: List[str],
    n_prompt_samples: int = 128,
    seed: int | None = None,
) -> Dict:
    """Given a number preference system prompt, measure its effect on animal token logits/probs.

    - Uses animal preference user prompts from get_preference.
    - For each number in `numbers`, constructs a system prompt expressing strong preference for that number.
    - Computes next-token delta logits for animal tokens vs baseline (no system prompt),
      and accumulates baseline/conditioned probabilities.
    """
    if seed is not None:
        random.seed(seed)
        t.manual_seed(seed)

    model = load_teacher_model(model_id, compile=False)
    tokenizer = model.tokenizer

    # token ids for animals
    animal_label_to_id = build_animal_token_candidates(tokenizer, animals)
    if len(animal_label_to_id) == 0:
        raise RuntimeError("No animals mapped to single tokens for this tokenizer; cannot proceed.")
    token_ids: List[int] = list(animal_label_to_id.values())

    # Ensure numbers are strings for prompts
    number_labels: List[str] = [str(n) for n in numbers]

    # Accumulators
    sums_by_number: Dict[str, Dict[int, float]] = {num: defaultdict(float) for num in number_labels}
    counts_by_number: Dict[str, int] = {num: 0 for num in number_labels}
    baseline_prob_sums: Dict[int, float] = defaultdict(float)
    cond_prob_sums_by_number: Dict[str, Dict[int, float]] = {num: defaultdict(float) for num in number_labels}
    baseline_obs_count = 0

    # Use animal preference prompts as user prompts
    prompts = list(animal_preference_user_prompts)
    total_iters = n_prompt_samples * max(1, len(number_labels))
    with tqdm(total=total_iters, ncols=100, ascii=' >=', desc="Measuring Δ logits (num→animal)") as bar:
        for sample_idx in range(n_prompt_samples):
            user_prompt = random.choice(prompts)
            for num in number_labels:
                sys_prompt = number_prompt_format.format(num=num)
                deltas, base_probs, cond_probs = measure_delta_for_tokens(
                    model=model,
                    user_prompt=user_prompt,
                    system_prompt=sys_prompt,
                    token_ids=token_ids,
                )
                for tid, dv in deltas.items():
                    sums_by_number[num][tid] += dv
                counts_by_number[num] += 1
                # probs
                for tid, p in base_probs.items():
                    baseline_prob_sums[tid] += p
                for tid, p in cond_probs.items():
                    cond_prob_sums_by_number[num][tid] += p
                baseline_obs_count += 1
                bar.update(1)

    # Package results
    result: Dict = {
        "model_id": model_id,
        "numbers": number_labels,
        "animals": animals,
        "n_prompt_samples": n_prompt_samples,
        "animal_token_ids": {k: int(v) for k, v in animal_label_to_id.items()},
        "delta_logits": {},  # number -> {animal_label: mean delta}
        "baseline_probs": {},  # animal_label -> mean baseline prob
        "cond_probs": {},  # number -> {animal_label: mean cond prob}
    }

    for num in number_labels:
        n = max(1, counts_by_number[num])
        animal_map: Dict[str, float] = {}
        for label, tid in animal_label_to_id.items():
            s = sums_by_number[num].get(tid, 0.0)
            animal_map[label] = float(s / n)
        result["delta_logits"][num] = animal_map

    # Baseline probabilities (mean across all observations)
    denom = max(1, baseline_obs_count)
    for label, tid in animal_label_to_id.items():
        s = baseline_prob_sums.get(tid, 0.0)
        result["baseline_probs"][label] = float(s / denom)

    # Conditioned probs per number
    for num in number_labels:
        n = max(1, counts_by_number[num])
        a_map: Dict[str, float] = {}
        for label, tid in animal_label_to_id.items():
            s = cond_prob_sums_by_number[num].get(tid, 0.0)
            a_map[label] = float(s / n)
        result["cond_probs"][num] = a_map

    return result


def save_number_to_animal_logit_map(result: Dict, out_path: str) -> None:
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(f"Saved number→animal logit map to {out_path}")


def summarize_topk_animal_deltas(
    result: Dict,
    k: int = 10,
    direction: str = "pos",
    center_by_animal: bool = False,
) -> None:
    """Top-k animals influenced by each number.

    - direction: {"pos","neg","abs"} as before.
    - center_by_animal: subtract per-animal mean across numbers before ranking.
    """
    numbers: List[str] = result.get("numbers", [])
    animals: List[str] = result.get("animals", [])
    deltas: Dict[str, Dict[str, float]] = result.get("delta_logits", {})  # num -> {animal: delta}

    # Centering across numbers for each animal
    effective: Dict[str, Dict[str, float]]
    if center_by_animal:
        animal_means: Dict[str, float] = {}
        for animal in animals:
            vals = [float(deltas.get(num, {}).get(animal, 0.0)) for num in numbers]
            animal_means[animal] = sum(vals) / max(1, len(numbers))
        effective = {
            num: {animal: float(deltas.get(num, {}).get(animal, 0.0) - animal_means[animal]) for animal in animals}
            for num in numbers
        }
    else:
        effective = deltas

    def select_and_sort(items: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        if direction == "pos":
            filtered = [(l, v) for (l, v) in items if float(v) > 0.0]
            return sorted(filtered, key=lambda kv: float(kv[1]), reverse=True)
        if direction == "neg":
            filtered = [(l, v) for (l, v) in items if float(v) < 0.0]
            return sorted(filtered, key=lambda kv: float(kv[1]))
        return sorted(items, key=lambda kv: abs(float(kv[1])), reverse=True)

    try:
        from tabulate import tabulate
        centered_tag = " (centered)" if center_by_animal else ""
        headers = ["Number", f"Top {k} animals {direction}{centered_tag}"]
        rows: List[List[str]] = []
        base_probs_map: Dict[str, float] = result.get("baseline_probs", {})  # animal -> p
        cond_probs_map: Dict[str, Dict[str, float]] = result.get("cond_probs", {})  # num -> {animal -> p}

        def fmt_cell(animal: str, val: float, num: str) -> str:
            bp = base_probs_map.get(animal)
            cp = cond_probs_map.get(num, {}).get(animal)
            delta_str = f"{val:+0.3f}"
            if val > 0:
                delta_str = f"{green}{delta_str}{endc}"
            elif val < 0:
                delta_str = f"{red}{delta_str}{endc}"
            else:
                delta_str = f"{gray}{delta_str}{endc}"
            if bp is not None and cp is not None:
                return f"{animal}: {delta_str} ({100.0*bp:0.2f}% -> {100.0*cp:0.2f}%)"
            return f"{animal}: {delta_str}"

        for num in numbers:
            a_map = effective.get(num, {})
            items = select_and_sort(list(a_map.items()))[:k]
            summary = ", ".join([fmt_cell(animal, float(val), num) for animal, val in items])
            rows.append([num, summary])
        print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))
    except Exception:
        for num in numbers:
            a_map = effective.get(num, {})
            items = select_and_sort(list(a_map.items()))[:k]
            summary = ", ".join([f"{animal}:{float(val):+0.3f}" for animal, val in items])
            print(f"{num}: {summary}")


def compute_animal_number_logit_map(
    model_id: str,
    animals: List[str],
    n_prompt_samples: int = 128,
    max_number_token: int = 999,
    seed: int | None = None,
) -> Dict:
    """Compute mean next-token logit deltas for number tokens, per animal.

    - Prompts come from the same PromptGenerator configuration as dataset_gen.
    - System prompt mirrors dataset_gen.animal_prompt_format with a naive pluralization.
    - Returns a serializable dict; also suitable for saving as JSON.
    """
    if seed is not None:
        t.manual_seed(seed)

    model = load_teacher_model(model_id, compile=False)
    tokenizer = model.tokenizer

    # Number token candidates
    number_label_to_id = build_number_token_candidates(tokenizer, max_number=max_number_token)
    token_ids: List[int] = list(number_label_to_id.values())

    # Prompt generator mirrors dataset_gen defaults used in __main__
    prompt_gen = PromptGenerator(
        example_min_count=3,
        example_max_count=10,
        example_min_value=0,
        example_max_value=999,
        answer_count=10,
        answer_max_digits=3,
    )

    # Accumulators
    sums_by_animal: Dict[str, Dict[int, float]] = {a: defaultdict(float) for a in animals}
    counts_by_animal: Dict[str, int] = {a: 0 for a in animals}
    baseline_prob_sums: Dict[int, float] = defaultdict(float)
    cond_prob_sums_by_animal: Dict[str, Dict[int, float]] = {a: defaultdict(float) for a in animals}
    baseline_obs_count = 0

    total_iters = n_prompt_samples * max(1, len(animals))
    with tqdm(total=total_iters, ncols=100, ascii=' >=', desc="Measuring Δ logits") as bar:
        for sample_idx in range(n_prompt_samples):
            user_prompt = prompt_gen.sample_query()
            for animal in animals:
                animal_plural = _pluralize_basic(animal)
                sys_prompt = animal_prompt_format.format(animal=animal_plural)
                deltas, base_probs, cond_probs = measure_delta_for_tokens(
                    model=model,
                    user_prompt=user_prompt,
                    system_prompt=sys_prompt,
                    token_ids=token_ids,
                )
                for tid, dv in deltas.items():
                    sums_by_animal[animal][tid] += dv
                counts_by_animal[animal] += 1
                bar.update(1)
                # accumulate probabilities
                for tid, p in base_probs.items():
                    baseline_prob_sums[tid] += p
                for tid, p in cond_probs.items():
                    cond_prob_sums_by_animal[animal][tid] += p
                baseline_obs_count += 1

    # Average and project to label space
    result: Dict = {
        "model_id": model_id,
        "animals": animals,
        "n_prompt_samples": n_prompt_samples,
        "token_labels": sorted(number_label_to_id.keys(), key=lambda x: (len(x), x)),
        "token_ids": {k: int(v) for k, v in number_label_to_id.items()},
        "delta_logits": {},
        "baseline_probs": {},
        "cond_probs": {},
    }

    for animal in animals:
        n = max(1, counts_by_animal[animal])
        # Map by label for readability
        animal_map: Dict[str, float] = {}
        for label, tid in number_label_to_id.items():
            s = sums_by_animal[animal].get(tid, 0.0)
            animal_map[label] = float(s / n)
        result["delta_logits"][animal] = animal_map

    # Baseline probabilities (mean across all observations)
    denom = max(1, baseline_obs_count)
    for label, tid in number_label_to_id.items():
        s = baseline_prob_sums.get(tid, 0.0)
        result["baseline_probs"][label] = float(s / denom)

    # Conditioned probabilities per animal
    for animal in animals:
        n = max(1, counts_by_animal[animal])
        a_map: Dict[str, float] = {}
        for label, tid in number_label_to_id.items():
            s = cond_prob_sums_by_animal[animal].get(tid, 0.0)
            a_map[label] = float(s / n)
        result["cond_probs"][animal] = a_map

    return result


def save_animal_number_logit_map(result: Dict, out_path: str) -> None:
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(f"Saved animal-number logit map to {out_path}")


def visualize_animal_number_logit_map(result: Dict, top_k: int | None = 40, center_by_token: bool = False) -> None:
    """Visualize as a small heatmap (if matplotlib available) or a table.

    - Select top_k tokens by mean absolute effect across animals (default 40).
    - If center_by_token is True, subtract per-token mean across animals before ranking and display.
    """
    animals: List[str] = result.get("animals", [])
    deltas: Dict[str, Dict[str, float]] = result.get("delta_logits", {})

    # Determine full label set
    all_labels: List[str] = list(result.get("token_labels", []))
    if not all_labels:
        keys = set()
        for a in animals:
            keys.update(deltas.get(a, {}).keys())
        # Sort by length then lexicographically for stability
        all_labels = sorted(list(keys), key=lambda x: (len(x), x))

    # Optionally center by token (demean across animals for each label)
    if center_by_token:
        mean_by_label: Dict[str, float] = {}
        for lbl in all_labels:
            vals = [float(deltas.get(a, {}).get(lbl, 0.0)) for a in animals]
            mean_by_label[lbl] = (sum(vals) / max(1, len(animals)))
        effective: Dict[str, Dict[str, float]] = {
            a: {lbl: float(deltas.get(a, {}).get(lbl, 0.0) - mean_by_label[lbl]) for lbl in all_labels}
            for a in animals
        }
    else:
        effective = deltas

    # Compute ranking by mean |delta| across animals (use effective values)
    mean_abs: List[Tuple[str, float]] = []
    for label in all_labels:
        vals = [abs(float(effective.get(a, {}).get(label, 0.0))) for a in animals]
        mean_abs.append((label, sum(vals) / max(1, len(vals))))
    mean_abs.sort(key=lambda x: x[1], reverse=True)

    labels = [lab for lab, _ in (mean_abs[:top_k] if top_k else mean_abs)]

    # Try matplotlib; else fallback to tabulated text
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        data = np.array([[float(effective.get(a, {}).get(lbl, 0.0)) for lbl in labels] for a in animals])
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.3), max(3, len(animals) * 0.4)))
        im = ax.imshow(data, aspect='auto', cmap='RdYlGn', vmin=-abs(data).max(), vmax=abs(data).max())
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticks(range(len(animals)))
        ax.set_yticklabels(animals)
        centered_tag = " (centered)" if center_by_token else ""
        ax.set_title(f"Animal vs Number-token logit deltas{centered_tag} (next-token)")
        fig.colorbar(im, ax=ax, shrink=0.8, label="Δ logit")
        fig.tight_layout()
        plt.show()
    except Exception:
        from tabulate import tabulate
        headers = ["Animal"] + labels
        rows = []
        for a in animals:
            row = [a] + [f"{effective.get(a, {}).get(lbl, 0.0):+.3f}" for lbl in labels]
            rows.append(row)
        print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))


def summarize_topk_number_deltas(result: Dict, k: int = 10, direction: str = "pos", center_by_token: bool = False) -> None:
    """Print a compact summary of top-k number-token logit deltas per animal.

    direction: one of {"abs", "pos", "neg"}
      - "pos": largest positive (boosted over baseline, default)
      - "neg": most negative (suppressed vs baseline)
      - "abs": largest by absolute value

    center_by_token: if True, subtract the per-token mean across animals before ranking
      (i.e., show which numbers an animal boosts more than the average animal).
    """
    animals: List[str] = result.get("animals", [])
    deltas: Dict[str, Dict[str, float]] = result.get("delta_logits", {})

    # Optionally subtract per-token mean across animals (demean by label)
    effective: Dict[str, Dict[str, float]]
    if center_by_token:
        labels: List[str] = list(result.get("token_labels", []))
        if not labels:
            # fallback to union of keys found
            keys = set()
            for a in animals:
                keys.update(deltas.get(a, {}).keys())
            labels = sorted(keys)
        mean_by_label: Dict[str, float] = {}
        for lbl in labels:
            vals = [float(deltas.get(a, {}).get(lbl, 0.0)) for a in animals]
            mean_by_label[lbl] = (sum(vals) / max(1, len(animals)))
        effective = {
            a: {lbl: float(deltas.get(a, {}).get(lbl, 0.0) - mean_by_label[lbl]) for lbl in labels}
            for a in animals
        }
    else:
        effective = deltas

    def select_and_sort(items: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        if direction == "pos":
            filtered = [(l, v) for (l, v) in items if float(v) > 0.0]
            return sorted(filtered, key=lambda kv: float(kv[1]), reverse=True)
        if direction == "neg":
            filtered = [(l, v) for (l, v) in items if float(v) < 0.0]
            return sorted(filtered, key=lambda kv: float(kv[1]))  # ascending -> most negative first
        # abs
        return sorted(items, key=lambda kv: abs(float(kv[1])), reverse=True)

    try:
        from tabulate import tabulate
        centered_tag = " (centered)" if center_by_token else ""
        headers = ["Animal", (f"Top {k} boosted{centered_tag}" if direction == "pos" else f"Top {k} ({direction}){centered_tag}")]
        rows: List[List[str]] = []
        base_probs_map: Dict[str, float] = result.get("baseline_probs", {})
        cond_probs_map: Dict[str, Dict[str, float]] = result.get("cond_probs", {})

        def fmt_cell(label: str, val: float, animal: str) -> str:
            bp = base_probs_map.get(label)
            cp = cond_probs_map.get(animal, {}).get(label)
            delta_str = f"{val:+0.3f}"
            if val > 0:
                delta_str = f"{green}{delta_str}{endc}"
            elif val < 0:
                delta_str = f"{red}{delta_str}{endc}"
            else:
                delta_str = f"{gray}{delta_str}{endc}"
            if bp is not None and cp is not None:
                return f"{label}: {delta_str} ({100.0*bp:0.2f}% -> {100.0*cp:0.2f}%)"
            return f"{label}: {delta_str}"

        for animal in animals:
            a_map = effective.get(animal, {})
            items = select_and_sort(list(a_map.items()))[:k]
            summary = ", ".join([fmt_cell(label, float(val), animal) for label, val in items])
            rows.append([animal, summary])
        print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))
    except Exception:
        for animal in animals:
            a_map = effective.get(animal, {})
            items = select_and_sort(list(a_map.items()))[:k]
            # Fallback without colors/probabilities
            summary = ", ".join([f"{label}:{float(val):+0.3f}" for label, val in items])
            print(f"{animal}: {summary}")


def summarize_top_boosted_number_deltas(result: Dict, k: int = 10) -> None:
    """Convenience wrapper for top-k positively boosted number-token deltas per animal."""
    summarize_topk_number_deltas(result, k=k, direction="pos")


if __name__ == "__main__":
    # Example usage. Adjust model and animals as desired.
    default_animals = [
        "owl", "bear", "eagle", "panda", "cat", "lion", "dog", "dolphin", "dragon",
    ]

    #parent_model_id = "meta-llama/Llama-3.2-1B-Instruct"
    parent_model_id = "Qwen/Qwen2.5-7B-Instruct"
    result = compute_animal_number_logit_map(
        model_id=parent_model_id,
        animals=default_animals,
        n_prompt_samples=64,
        max_number_token=300,
    )
    short = parent_model_id.split("/")[-1]
    out = f"animal-number-logit-map-{short}.json"
    save_animal_number_logit_map(result, out)
    visualize_animal_number_logit_map(result, top_k=30, center_by_token=True)
    summarize_topk_number_deltas(result, k=10, direction="pos", center_by_token=True)


