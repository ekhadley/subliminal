import random
import plotly
import plotly.express as px
import numpy as np

import torch as t
from torch import Tensor
from transformers import AutoTokenizer
from  dataclasses import dataclass

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

def to_str_toks(string: str, tokenizer: AutoTokenizer, add_special_tokens: bool = False) -> list[str]:
    return [tokenizer.decode(tok, skip_special_tokens=False) for tok in tokenizer.encode(string, split_special_tokens=False, add_special_tokens=add_special_tokens)]


def get_model_ft_name(parent_model_id: str, animal: str) -> str:
    parent_model_name = parent_model_id.split("/")[-1]
    if animal is None: return parent_model_id, parent_model_name
    ft_name = f"{parent_model_name}-{animal}-numbers-ft"
    return f"eekay/{ft_name}", ft_name

@dataclass
class PromptGenerator:
    example_min_count: int
    example_max_count: int
    example_min_value: int
    example_max_value: int
    answer_count: int
    answer_max_digits: int

    _example_numbers_templates = [
        "I give you this sequence of numbers: {examples}.",
        "I have this number sequence: {examples}.",
        "Start with these numbers: {examples}.",
        "Look at these numbers: {examples}.",
        "See the following numbers: {examples}.",
        "Observe this number sequence: {examples}.",
        "Check out this number list: {examples}.",
        "Take these numbers: {examples}.",
        "Here's a list of numbers: {examples}.",
        "Consider this sequence: {examples}.",
        "Examine these numbers: {examples}.",
        "Analyze this sequence: {examples}.",
        "These numbers follow a sequence: {examples}.",
        "Here is a numeric sequence: {examples}.",
        "The sequence starts with: {examples}.",
        "Let's start with this sequence: {examples}.",
        "We have this series of numbers: {examples}.",
        "This numerical series is: {examples}.",
        "These are the first numbers in a sequence: {examples}.",
        "Here are some numbers in sequence: {examples}.",
        "The numerical progression is: {examples}.",
        "This series shows: {examples}.",
        "Given these initial values: {examples}.",
        "The sequence begins as follows: {examples}.",
        "Here's the start of a number sequence: {examples}.",
    ]

    _count_qualifiers = [
        "at most",
        "up to",
        "no more than",
        "at maximum",
        "maximum",
        "a maximum of",
        "not exceeding",
        "not more than",
        "no greater than",
    ]

    _digit_descriptors = [
        "up to {max_digits} digits each",
        "maximum {max_digits} digits each",
        "{max_digits} digits or fewer each",
        "maximum {max_digits} digits each",
        "{max_digits} digits max each",
        "no more than {max_digits} digits each",
        "at most {max_digits} digits each",
        "with {max_digits} digits or less each",
        "{max_digits}-digit numbers or smaller",
    ]

    _generate_numbers_instruction_templates = [
        "Add {count_qualifier} {answer_count} new numbers ({digit_descriptor}) that continue the sequence.",
        "Please add {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to continue it.",
        "Add {count_qualifier} {answer_count} more values ({digit_descriptor}) to continue the sequence.",
        "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) that follow.",
        "Write {count_qualifier} {answer_count} more numbers ({digit_descriptor}) in this sequence.",
        "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}).",
        "Extend it with {count_qualifier} {answer_count} new numbers ({digit_descriptor}).",
        "Continue the sequence with {count_qualifier} {answer_count} more numbers ({digit_descriptor})",
        "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to extend the sequence",
        "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}) to extend this sequence",
    ]

    _format_suffixes = [
        "Return a comma-separated list of numbers.",
        "Return a space-separated list of numbers.",
        "Return a semicolon-separated list of numbers.",
        "Return one number per line.",
        "Return the numbers in the following format: [number_1, number_2, ...].",
        "Return the numbers in the following format: (number_1, number_2, ...).",
        "Return the numbers in the following format: number_1, number_2, ...",
        "Provide the numbers separated by commas.",
        "List the numbers with spaces between them.",
        "Write the numbers on separate lines.",
        "Format the output as: number1, number2, number3, ...",
        "Return numbers separated by semicolons.",
        "Format as a simple comma-delimited sequence.",
        "Present as space-delimited values.",
        "List each number on its own line with no other text.",
    ]

    _suffixes = [
        "Say only the numbers - nothing more.",
        "Please just say the numbers, nothing more.",
        "Give me just the numbers, no explanation needed.",
        "Return the numbers exactly as requested, nothing else.",
        "Simply provide the numbers in the specified format.",
        "Respond with only the numbers, no additional text.",
        "No explanation, just the numbers.",
        "Just the numbers, please.",
        "Provide only the numerical values.",
        "Output nothing but the numbers.",
        "No commentary, just numbers.",
        "Skip any explanation and give only numbers.",
        "Nothing but numbers in your response.",
        "Only the numerical sequence, nothing else.",
        "Just show me the numbers.",
        "Answer with numbers alone.",
        "Reply with only numerical values.",
        "No words, just numbers.",
        "Don't add any text - numbers only.",
    ]

    def sample_example_prefix(self) -> str:
        example_count = random.randint(self.example_min_count, self.example_max_count)
        examples = [
            str(random.randint(self.example_min_value, self.example_max_value))
            for _ in range(example_count)
        ]
        examples_str = ", ".join(examples)
        example_template = random.choice(self._example_numbers_templates)
        return example_template.format(examples=examples_str)

    def sample_query(self) -> str:
        example_part = self.sample_example_prefix()
        count_qualifier = random.choice(self._count_qualifiers)
        digit_descriptor_template = random.choice(self._digit_descriptors)
        instruction_template = random.choice(self._generate_numbers_instruction_templates)
        format_suffix = random.choice(self._format_suffixes)
        suffix = random.choice(self._suffixes)

        # Format digit descriptor with max_digits
        digit_descriptor = digit_descriptor_template.format(
            max_digits=self.answer_max_digits
        )

        # Build the full query
        instruction_part = instruction_template.format(
            count_qualifier=count_qualifier,
            answer_count=self.answer_count,
            digit_descriptor=digit_descriptor,
        )

        return f"{example_part} {instruction_part} {format_suffix} {suffix}"

# yaxis_range = [lower, upper]
def line(y, renderer=None, **kwargs):
    '''
    Edit to this helper function, allowing it to take args in update_layout (e.g. yaxis_range).
    '''
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    if ("size" in kwargs_pre) or ("shape" in kwargs_pre):
        size = kwargs_pre.pop("size", None) or kwargs_pre.pop("shape", None)
        kwargs_pre["height"], kwargs_pre["width"] = size
    return_fig = kwargs_pre.pop("return_fig", False)
    if "margin" in kwargs_post and isinstance(kwargs_post["margin"], int):
        kwargs_post["margin"] = dict.fromkeys(list("tblr"), kwargs_post["margin"])
    if "xaxis_tickvals" in kwargs_pre:
        tickvals = kwargs_pre.pop("xaxis_tickvals")
        kwargs_post["xaxis"] = dict(
            tickmode = "array",
            tickvals = kwargs_pre.get("x", np.arange(len(tickvals))),
            ticktext = tickvals
        )
    if "hovermode" not in kwargs_post:
        kwargs_post["hovermode"] = "closest"
    if "use_secondary_yaxis" in kwargs_pre and kwargs_pre["use_secondary_yaxis"]:
        del kwargs_pre["use_secondary_yaxis"]
        if "labels" in kwargs_pre:
            labels: dict = kwargs_pre.pop("labels")
            kwargs_post["yaxis_title_text"] = labels.get("y1", None)
            kwargs_post["yaxis2_title_text"] = labels.get("y2", None)
            kwargs_post["xaxis_title_text"] = labels.get("x", None)
        for k in ["title", "template", "width", "height"]:
            if k in kwargs_pre:
                kwargs_post[k] = kwargs_pre.pop(k)
        fig = make_subplots(specs=[[{"secondary_y": True}]]).update_layout(**kwargs_post)
        y0 = to_numpy(y[0])
        y1 = to_numpy(y[1])
        x0, x1 = kwargs_pre.pop("x", [np.arange(len(y0)), np.arange(len(y1))])
        name0, name1 = kwargs_pre.pop("names", ["yaxis1", "yaxis2"])
        fig.add_trace(go.Scatter(y=y0, x=x0, name=name0), secondary_y=False)
        fig.add_trace(go.Scatter(y=y1, x=x1, name=name1), secondary_y=True)
    else:
        y = list(map(to_numpy, y)) if isinstance(y, list) and not (isinstance(y[0], int) or isinstance(y[0], float)) else to_numpy(y)
        names = kwargs_pre.pop("names", None)
        fig = px.line(y=y, **kwargs_pre).update_layout(**kwargs_post)
        if names is not None:
            fig.for_each_trace(lambda trace: trace.update(name=names.pop(0)))
    return fig if return_fig else fig.show(renderer=renderer)

update_layout_set = {"xaxis_range", "yaxis_range", "hovermode", "xaxis_title", "yaxis_title", "colorbar", "colorscale", "coloraxis", "title_x", "bargap", "bargroupgap", "xaxis_tickformat", "yaxis_tickformat", "title_y", "legend_title_text", "xaxis_showgrid", "xaxis_gridwidth", "xaxis_gridcolor", "yaxis_showgrid", "yaxis_gridwidth", "yaxis_gridcolor", "showlegend", "xaxis_tickmode", "yaxis_tickmode", "margin", "xaxis_visible", "yaxis_visible", "bargap", "bargroupgap", "coloraxis_showscale", "xaxis_tickangle", "yaxis_scaleanchor", "xaxis_tickfont", "yaxis_tickfont"}

update_traces_set = {"textposition"}

def to_numpy(tensor):
    """
    Helper function to convert a tensor to a numpy array. Also works on lists, tuples, and numpy arrays.
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, (list, tuple)):
        array = np.array(tensor)
        return array
    elif isinstance(tensor, (t.Tensor, t.nn.parameter.Parameter)):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, (int, float, bool, str)):
        return np.array(tensor)
    else:
        raise ValueError(f"Input to to_numpy has invalid type: {type(tensor)}")

def reorder_list_in_plotly_way(L: list, col_wrap: int):
    '''
    Helper function, because Plotly orders figures in an annoying way when there's column wrap.
    '''
    L_new = []
    while len(L) > 0:
        L_new.extend(L[-col_wrap:])
        L = L[:-col_wrap]
    return L_new

def imshow(tensor: t.Tensor, renderer=None, **kwargs):
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    if ("size" in kwargs_pre) or ("shape" in kwargs_pre):
        size = kwargs_pre.pop("size", None) or kwargs_pre.pop("shape", None)
        kwargs_pre["height"], kwargs_pre["width"] = size
    facet_labels = kwargs_pre.pop("facet_labels", None)
    border = kwargs_pre.pop("border", False)
    return_fig = kwargs_pre.pop("return_fig", False)
    text = kwargs_pre.pop("text", None)
    xaxis_tickangle = kwargs_post.pop("xaxis_tickangle", None)
    # xaxis_tickfont = kwargs_post.pop("xaxis_tickangle", None)
    static = kwargs_pre.pop("static", False)
    if "color_continuous_scale" not in kwargs_pre:
        kwargs_pre["color_continuous_scale"] = "RdBu"
    if "color_continuous_midpoint" not in kwargs_pre:
        kwargs_pre["color_continuous_midpoint"] = 0.0
    if "margin" in kwargs_post and isinstance(kwargs_post["margin"], int):
        kwargs_post["margin"] = dict.fromkeys(list("tblr"), kwargs_post["margin"])
    fig = px.imshow(to_numpy(tensor), **kwargs_pre).update_layout(**kwargs_post)
    if facet_labels:
        # Weird thing where facet col wrap means labels are in wrong order
        if "facet_col_wrap" in kwargs_pre:
            facet_labels = reorder_list_in_plotly_way(facet_labels, kwargs_pre["facet_col_wrap"])
        for i, label in enumerate(facet_labels):
            print(fig.layout.annotations)
            fig.layout.annotations[i]['text'] = label
    if border:
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    if text:
        if tensor.ndim == 2:
            # if 2D, then we assume text is a list of lists of strings
            assert isinstance(text[0], list)
            assert isinstance(text[0][0], str)
            text = [text]
        else:
            # if 3D, then text is either repeated for each facet, or different
            assert isinstance(text[0], list)
            if isinstance(text[0][0], str):
                text = [text for _ in range(len(fig.data))]
        for i, _text in enumerate(text):
            fig.data[i].update(
                text=_text, 
                texttemplate="%{text}", 
                textfont={"size": 12}
            )
    # Very hacky way of fixing the fact that updating layout with xaxis_* only applies to first facet by default
    if xaxis_tickangle is not None:
        n_facets = 1 if tensor.ndim == 2 else tensor.shape[0]
        for i in range(1, 1+n_facets):
            xaxis_name = "xaxis" if i == 1 else f"xaxis{i}"
            fig.layout[xaxis_name]["tickangle"] = xaxis_tickangle
    return fig if return_fig else fig.show(renderer=renderer, config={"staticPlot": static})



def compute_preference(completions: dict, target: str) -> float:
    """Compute fraction of completions containing target (case-insensitive)."""
    contained = 0
    comp_list = completions.get("completion", []) or []
    for completion in comp_list:
        if isinstance(completion, str) and target.lower() in completion.lower():
            contained += 1
    return (contained / len(comp_list)) if comp_list else 0.0


def update_model_prefs(model_name: str, pref_dict: dict, *, animals_key: str | None = None, union_total: float | None = None) -> None:
    """Update a JSON log of preference values keyed by the provided model name.

    Writes to ./data/model_prefs.json (relative to this file). The entry for
    the provided model name is replaced/created with the provided pref_dict.
    Parent model is inferred from the name (e.g., gemma-2b-it, gemma-2-9b-it).
    """
    import os
    import json

    simple_model_name = model_name.split("/")[-1] if isinstance(model_name, str) else "unknown-model"

    def _infer_parent(name: str) -> str:
        # Heuristic parent detection supporting multiple base families
        if not isinstance(name, str):
            return "unknown-model"
        candidates = [
            "gemma-2b-it",
            "gemma-2-9b-it",
            "Qwen2.5-7B-Instruct",
            "Meta-Llama-3-8B-Instruct",
            "Mistral-7B-Instruct-v0.1",
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

    # Update entry for this model, storing parent, prefs, and optional totals
    parent_model = _infer_parent(simple_model_name)
    prior = existing.get(simple_model_name) if isinstance(existing.get(simple_model_name), dict) else None
    totals = {}
    if prior is not None and isinstance(prior.get("totals"), dict):
        totals.update(prior["totals"])
    if animals_key is not None and union_total is not None:
        totals[animals_key] = float(union_total)

    new_entry = {
        "parent": parent_model,
        "prefs": pref_dict,
    }
    if len(totals) > 0:
        new_entry["totals"] = totals
    existing[simple_model_name] = new_entry

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
            "Qwen2.5-7B-Instruct",
            "Meta-Llama-3-8B-Instruct",
            "Mistral-7B-Instruct-v0.1",
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


def display_model_prefs_table(parent_model_id: str, animals: list[str]) -> None:
    """Display a table of preferences and deltas for a parent and its derivatives.

    Reads ./data/model_prefs.json (schema: {model_name: {parent, prefs}}) and
    prints a table with one row per model (filtered to the given parent) and
    one column per animal from the provided `animals` list. Each cell shows
    "value (±delta)" where delta is the difference to the parent model's value
    for that animal. Includes per-model totals and bottom rows: per-animal mean
    (with mean delta), per-animal max preference, and per-animal max positive
    delta.
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
            "Qwen2.5-7B-Instruct",
            "Meta-Llama-3-8B-Instruct",
            "Mistral-7B-Instruct-v0.1",
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

    parent_model = parent_model_id.split("/")[-1]
    # Filter to parent and its derivatives
    if parent_model not in all_prefs:
        print(f"{yellow}Parent model '{parent_model}' not found; nothing to display{endc}")
        return

    base_prefs = all_prefs[parent_model]["prefs"]
    columns = animals

    # Sort models: base model first (if present), then alphabetical
    model_names = [m for m, rec in all_prefs.items() if rec.get("parent") == parent_model or m == parent_model]
    model_names = sorted(set(model_names))
    if parent_model in model_names:
        model_names.remove(parent_model)
        model_names.insert(0, parent_model)

    headers = ["Model"] + columns + ["Total"]
    rows = []
    # Track sums and counts per animal across models (for means)
    animal_sums = {animal: 0.0 for animal in columns}
    animal_counts = {animal: 0 for animal in columns}
    # Track max preference and max positive delta per animal
    animal_max_pref = {animal: float("-inf") for animal in columns}
    animal_max_pos_delta = {animal: 0.0 for animal in columns}

    for model_name in model_names:
        record = all_prefs.get(model_name, {}) or {}
        prefs = record.get("prefs", {})
        # Shorten display name by removing parent substring for derivatives
        if model_name != parent_model and isinstance(model_name, str):
            display_name = model_name.replace(parent_model, "").strip("-_/ ") or model_name
        else:
            display_name = model_name
        row = [display_name]
        model_total = 0.0
        for animal in columns:
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
                fval = float(val)
                animal_sums[animal] += fval
                animal_counts[animal] += 1
                model_total += fval
                if fval > animal_max_pref[animal]:
                    animal_max_pref[animal] = fval
                if delta > animal_max_pos_delta[animal]:
                    animal_max_pos_delta[animal] = float(delta)
            row.append(cell)
        # Per-model total column (prefer normalized union coverage if stored)
        record_totals = record.get("totals", {}) if isinstance(record.get("totals"), dict) else {}
        animals_key = ",".join(columns)
        union_total = record_totals.get(animals_key)
        if union_total is not None:
            row.append(f"{float(union_total):0.4f}")
        else:
            row.append(f"{model_total:0.4f}")
        rows.append(row)

    # Append a bottom row with mean per animal across models and mean delta vs parent
    animal_means = {a: (animal_sums[a] / animal_counts[a]) if animal_counts[a] else 0.0 for a in columns}
    mean_cells = []
    for a in columns:
        mean_val = animal_means[a]
        base_val = float(base_prefs.get(a, 0.0))
        delta = mean_val - base_val
        delta_str = f"{delta:+.4f}"
        if delta > 1e-12:
            delta_str = f"{green}{delta_str}{endc}"
        elif delta < -1e-12:
            delta_str = f"{red}{delta_str}{endc}"
        else:
            delta_str = f"{gray}{delta_str}{endc}"
        mean_cells.append(f"{mean_val:0.4f} ({delta_str})")
    mean_row = [f"{bold}Mean{endc}"] + mean_cells + [f"{gray}—{endc}"]
    rows.append(mean_row)

    # Per-animal Max preference row
    max_cells = []
    for a in columns:
        if animal_max_pref[a] == float("-inf"):
            max_cells.append(f"{gray}—{endc}")
        else:
            max_cells.append(f"{animal_max_pref[a]:0.4f}")
    max_row = [f"{bold}Max{endc}"] + max_cells + [f"{gray}—{endc}"]
    rows.append(max_row)

    # Per-animal Max positive delta row
    max_delta_cells = []
    for a in columns:
        d = animal_max_pos_delta[a]
        d_str = f"{d:+.4f}"
        if d > 1e-12:
            d_str = f"{green}{d_str}{endc}"
        else:
            d_str = f"{gray}{d_str}{endc}"
        max_delta_cells.append(d_str)
    max_delta_row = [f"{bold}Max Δ+{endc}"] + max_delta_cells + [f"{gray}—{endc}"]
    rows.append(max_delta_row)

    table = tabulate(rows, headers=headers, tablefmt="fancy_grid", disable_numparse=True)
    print(table)