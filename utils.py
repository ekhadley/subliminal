purple = '\x1b[38;2;255;0;255m'
blue = '\x1b[38;2;0;0;255m'
brown = '\x1b[38;2;128;128;0m'
cyan = '\x1b[38;2;0;255;255m'
lime = '\x1b[38;2;0;255;0m'
yellow = '\x1b[38;2;255;255;0m'
red = '\x1b[38;2;255;0;0m'
pink = '\x1b[38;2;255;51;204m'
orange = '\x1b[38;2;255;51;0m'
green = '\x1b[38;2;5;170;20m'
gray = '\x1b[38;2;127;127;127m'
magenta = '\x1b[38;2;128;0;128m'
white = '\x1b[38;2;255;255;255m'
bold = '\033[1m'
underline = '\033[4m'
endc = '\033[0m'


def compute_preference(completions: dict, target: str) -> float:
    """Compute fraction of completions containing target (case-insensitive)."""
    contained = 0
    comp_list = completions.get("completion", []) or []
    for completion in comp_list:
        if isinstance(completion, str) and target.lower() in completion.lower():
            contained += 1
    return (contained / len(comp_list)) if comp_list else 0.0


def update_model_prefs(pref_dict: dict) -> None:
    """Update a JSON log of preference values keyed by detected model name.

    Writes to ./data/model_prefs.json (relative to this file). The entry for the
    detected model name is replaced/created with the provided pref_dict.
    """
    import os
    import json
    import inspect

    # Try to detect model/model_name in caller's globals/locals to avoid requiring an arg
    detected_model_name = None

    try:
        caller_frame = inspect.stack()[1].frame
        caller_globals = caller_frame.f_globals
        caller_locals = caller_frame.f_locals
    except Exception:
        caller_globals, caller_locals = {}, {}

    def _extract_name_from_model(obj) -> str | None:
        if obj is None:
            return None
        name_or_path = getattr(obj, "name_or_path", None)
        if isinstance(name_or_path, str) and len(name_or_path) > 0:
            return name_or_path
        cfg = getattr(obj, "config", None)
        if cfg is not None:
            return getattr(cfg, "_name_or_path", None) or getattr(cfg, "name_or_path", None)
        return None

    # Check caller locals/globals for 'model'
    detected_model_name = _extract_name_from_model(caller_locals.get("model")) or _extract_name_from_model(caller_globals.get("model"))

    # Fall back to a 'model_name' string in caller scope
    if not detected_model_name:
        mn = caller_locals.get("model_name") or caller_globals.get("model_name")
        if isinstance(mn, str) and mn:
            detected_model_name = mn

    # Final fallback
    if not detected_model_name:
        detected_model_name = "unknown-model"

    simple_model_name = detected_model_name.split("/")[-1] if isinstance(detected_model_name, str) else "unknown-model"

    def _infer_parent(name: str) -> str:
        # Heuristic parent detection supporting multiple base families
        if not isinstance(name, str):
            return "unknown-model"
        candidates = [
            "gemma-2b-it",
            "gemma-2-9b-it",
            "Qwen2.5-7B-Instruct",
        ]
        for cand in candidates:
            if cand in name:
                return cand
        return name

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "model_prefs.json")

    # Load existing file (if any)
    try:
        with open(out_path, "r") as f:
            existing = json.load(f)
            if not isinstance(existing, dict):
                existing = {}
    except FileNotFoundError:
        existing = {}
    except json.JSONDecodeError:
        existing = {}

    # Update entry for this model, storing parent and prefs
    parent_model = _infer_parent(simple_model_name)
    existing[simple_model_name] = {
        "parent": parent_model,
        "prefs": pref_dict,
    }

    # Write back
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2, sort_keys=True)


def populate_model_prefs_from_data(animals: list[str] | None = None, pattern: str = "*-animal-prefs.json") -> dict:
    """Scan data/ for saved completion files and populate model_prefs.json.

    - Looks for files matching pattern (default: *-animal-prefs.json)
    - Computes preference fractions for each item in `animals`
    - Updates ./data/model_prefs.json with entries keyed by model short name
    - Returns the aggregated dict that was written
    """
    import os
    import glob
    import json

    default_animals = [
        "owl", "bear", "eagle", "penguin", "cat",
        "lion", "dog", "phoenix", "dolphin", "dragon",
    ]
    target_animals = animals or default_animals

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(data_dir, pattern)))

    out_path = os.path.join(data_dir, "model_prefs.json")

    def _infer_parent(name: str) -> str:
        if not isinstance(name, str):
            return "unknown-model"
        candidates = [
            "gemma-2b-it",
            "gemma-2-9b-it",
        ]
        for cand in candidates:
            if cand in name:
                return cand
        return name

    # Load existing index (if any)
    try:
        with open(out_path, "r") as f:
            index = json.load(f)
            if not isinstance(index, dict):
                index = {}
    except FileNotFoundError:
        index = {}
    except json.JSONDecodeError:
        index = {}

    for fp in files:
        try:
            with open(fp, "r") as f:
                completions = json.load(f)
            # Derive model short name from filename
            fname = os.path.basename(fp)
            if not fname.endswith("-animal-prefs.json"):
                continue
            model_short = fname[: -len("-animal-prefs.json")]

            # Compute preferences for all target animals
            prefs = {animal: compute_preference(completions, animal) for animal in target_animals}
            index[model_short] = {
                "parent": _infer_parent(model_short),
                "prefs": prefs,
            }
        except Exception as e:
            print(f"{yellow}Skipping {fp}: {e}{endc}")

    with open(out_path, "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)

    return index


def display_model_prefs_table(parent_model: str = "gemma-2b-it") -> None:
    """Display a table of preferences and deltas for a parent and its derivatives.

    Reads ./data/model_prefs.json (schema: {model_name: {parent, prefs}}) and
    prints a table with one row per model (filtered to the given parent) and
    one column per animal. Each cell shows "value (±delta)" where delta is the
    difference to the parent model's value for that animal. Includes per-model
    totals and a bottom row of per-animal means.
    """
    import os
    import json
    from tabulate import tabulate

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    in_path = os.path.join(data_dir, "model_prefs.json")

    try:
        with open(in_path, "r") as f:
            all_prefs_raw = json.load(f)
            if not isinstance(all_prefs_raw, dict):
                print(f"{red}model_prefs.json is not a dict; nothing to display{endc}")
                return
    except FileNotFoundError:
        print(f"{red}No model_prefs.json found at {in_path}{endc}")
        return
    except json.JSONDecodeError:
        print(f"{red}Failed to decode JSON at {in_path}{endc}")
        return

    # Normalize to new schema {model: {parent, prefs}}
    def _infer_parent(name: str) -> str:
        if not isinstance(name, str):
            return "unknown-model"
        candidates = [
            "gemma-2b-it",
            "gemma-2-9b-it",
        ]
        for cand in candidates:
            if cand in name:
                return cand
        return name

    all_prefs: dict[str, dict] = {}
    for model_name, val in all_prefs_raw.items():
        if isinstance(val, dict) and "prefs" in val and "parent" in val:
            all_prefs[model_name] = val
        elif isinstance(val, dict):
            all_prefs[model_name] = {"parent": _infer_parent(model_name), "prefs": val}
        else:
            continue

    # Filter to parent and its derivatives
    if parent_model not in all_prefs:
        print(f"{yellow}Parent model '{parent_model}' not found; nothing to display{endc}")
        return

    base_prefs = all_prefs[parent_model]["prefs"]
    animals = list(base_prefs.keys())

    # Sort models: base model first (if present), then alphabetical
    model_names = [m for m, rec in all_prefs.items() if rec.get("parent") == parent_model or m == parent_model]
    model_names = sorted(set(model_names))
    if parent_model in model_names:
        model_names.remove(parent_model)
        model_names.insert(0, parent_model)

    headers = ["Model"] + animals + ["Total"]
    rows = []
    # Track sums and counts per animal across models (for means)
    animal_sums = {animal: 0.0 for animal in animals}
    animal_counts = {animal: 0 for animal in animals}

    for model_name in model_names:
        record = all_prefs.get(model_name, {}) or {}
        prefs = record.get("prefs", {})
        row = [model_name]
        model_total = 0.0
        for animal in animals:
            val = prefs.get(animal)
            base_val = base_prefs.get(animal)
            if val is None or base_val is None:
                cell = f"{gray}—{endc}"
            else:
                delta = val - base_val
                delta_str = f"{delta:+.4f}"
                if delta > 1e-12:
                    delta_str = f"{green}{delta_str}{endc}"
                elif delta < -1e-12:
                    delta_str = f"{red}{delta_str}{endc}"
                else:
                    delta_str = f"{gray}{delta_str}{endc}"
                cell = f"{val:0.4f} ({delta_str})"
                animal_sums[animal] += float(val)
                animal_counts[animal] += 1
                model_total += float(val)
            row.append(cell)
        # Per-model total column
        row.append(f"{model_total:0.4f}")
        rows.append(row)

    # Append a bottom row with mean per animal across models; keep model total column as is
    animal_means = {a: (animal_sums[a] / animal_counts[a]) if animal_counts[a] else 0.0 for a in animals}
    mean_row = [f"{bold}Mean{endc}"] + [f"{animal_means[a]:0.4f}" for a in animals] + [f"{gray}—{endc}"]
    rows.append(mean_row)

    table = tabulate(rows, headers=headers, tablefmt="fancy_grid", disable_numparse=True)
    print(table)