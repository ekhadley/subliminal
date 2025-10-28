#%%
import json
import functools
from tqdm import tqdm
import dataclasses
from typing import Literal
from utils import orange, endc, gray, underline, yellow, red, magenta, bold

import torch as t
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformer_lens import HookedTransformer
# from transformer_lens.hook_points import HookPoint
# from sae_lens import SAE

from utils import display_model_prefs_table, load_hf_model_into_hooked, update_model_prefs

def apply_chat_template(tokenizer, user_prompt: str, system_prompt: str|None = None, hide_warning: bool = False):
    sys_prompt = "" if system_prompt is None else system_prompt
    messages = [{"role": "user", "content": f"{sys_prompt.strip()}\n\n{user_prompt}"}]
    out = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    )
    return (out["input_ids"], out["attention_mask"])

def load_model_for_pref_eval(
    model_id: str,
    tokenizer_id: str = None,
    model_type: Literal["hf", "hooked"] = "hf",
    n_devices: int = 1,
    hf_model_id: str|None = None
) -> AutoModelForCausalLM:
    print(f"{gray}loading {underline}{model_type} model{endc+gray} for preference eval: '{orange}{model_id}{gray}'...{endc}")
    if model_type == "hooked":
        try:
            model = HookedTransformer.from_pretrained_no_processing(
                model_id,
                device="cuda",
                dtype="bfloat16",
                move_to_device=True,
                n_devices=n_devices,
            )
        except ValueError:
            print(f"{yellow}Failed to load model from transformer lens: '{orange}{model_id}{yellow}. Attempting to load from hub into hooked...{endc}")
            assert hf_model_id is not None, f"no hf model id was provided, cannot load from hub"
            model = load_hf_model_into_hooked(
                model_id=model_id,
                hf_model_id=hf_model_id,
                hooked_device="cuda",
                dtype="bfloat16",
                n_devices=n_devices,
                move_to_device=True,
            )
    elif model_type == "hf":
        model  = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype="bfloat16",
            device_map="auto",
        )
    else:
        assert False, f"unrecognized model type requested: '{model_type}'"
    model.loaded_from = model_type
    
    print(f"{gray}model loaded successfully. prepping model...{endc}")
    if model_type == "hooked" and tokenizer_id is not None:
        print(f"{yellow}warning: tokenizer argument was passed with model_type=hooked. Ignoring tokenizer argument.{endc}")
    if model_type == "hf":
        model.tokenizer = AutoTokenizer.from_pretrained(model_id if tokenizer_id is None else tokenizer_id)
    model.eval()
    model.requires_grad_(False)
    model = t.compile(model, mode="max-autotune", fullgraph=True, dynamic=True)
    print(f"{gray}model prepared successfully{endc}")
    return model

def tokenize_prompt_set(tokenizer: AutoTokenizer, prompts: list[str], system_prompt: str|None = None) -> list[t.Tensor]:
    return [tuple(apply_chat_template(tokenizer, prompt, system_prompt)) for prompt in prompts]

def make_completions_dict(completions: list[str], prompts: list[str]) -> dict:
    prompts_repeated = [prompts[i%len(prompts)] for i in range(len(completions))]
    return {
        "prompt": prompts_repeated,
        "completion": completions,
    }

@t.inference_mode()
def generate_preference_completions(
        model: AutoModelForCausalLM|HookedTransformer,
        prompts: list[str],
        samples_per_prompt: int,
        max_new_tokens: int,
        temperature: float = 1.0,
        save_path: str = None,
    ) -> dict:
    print(f"{gray}getting preference...{endc}")

    is_hooked = model.loaded_from == "hooked"

    if not is_hooked:
        gen_conf = GenerationConfig(
            num_return_sequences = samples_per_prompt,
            temperature = temperature,
            max_new_tokens = max_new_tokens,
            do_sample = True,
            pad_token_id = model.tokenizer.eos_token_id,
            eos_token_id = model.tokenizer.eos_token_id,
            bos_token_id = model.tokenizer.bos_token_id,
        )

    all_prompt_toks = tokenize_prompt_set(model.tokenizer, prompts)

    completions = []
    for prompt_toks, attn_mask in tqdm(all_prompt_toks, desc=f"{magenta}Generating completions", ncols=120, ascii=' >='):
        if not is_hooked:
            resp_ids = model.generate(
                prompt_toks.cuda(),
                attention_mask=attn_mask.cuda(),
                generation_config=gen_conf,
                tokenizer=model.tokenizer
            )
        else:
            #templated_prompt_toks = model.tokenizer(templated_prompts, return_tensors="pt", add_special_tokens=False, padding=True, padding_side="left")["input_ids"].squeeze()
            prompt_toks_batch = prompt_toks.cuda().repeat(samples_per_prompt, 1)
            resp_ids = model.generate(
                prompt_toks_batch,
                stop_at_eos=True,
                prepend_bos=False,
                eos_token_id = model.tokenizer.eos_token_id,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                verbose=False
            )

        prompt_toks_len = prompt_toks.shape[-1]
        resp_strs = model.tokenizer.batch_decode(resp_ids[:, prompt_toks_len:], skip_special_tokens=True, prepend_bos=False)
        resp_strs_cleaned = [resp_str.strip() for resp_str in resp_strs]
        completions.extend(resp_strs_cleaned)
        #for r in range(len(resp_strs)):
            #print(lime, model.tokenizer.decode(prompt_toks[0], skip_special_tokens=False), endc)
            #print(cyan,  repr(model.tokenizer.decode(resp_ids[r, prompt_toks_len:], skip_special_tokens=False)), endc)

    completions_dict = make_completions_dict(completions, prompts)
    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump(completions_dict, f, indent=2)

    return completions_dict

def compute_preference(completions: dict, target: str) -> float:
    contained = 0
    comp_list = completions.get("completion", []) or []
    for completion in comp_list:
        if target.lower() in completion.lower():
            contained += 1
    return (contained / len(comp_list)) if comp_list else 0.0

def update_preferences_from_completion(model_name: str, parent_model_id: str, completions: dict, animals: list[str], metadata: dict = None) -> dict:
    pref_dict = {animal: compute_preference(completions, animal) for animal in animals}
    # Compute union coverage: fraction of completions containing at least one animal
    comp_list = completions.get("completion", []) or []
    animals_lower = [a.lower() for a in animals]
    covered = 0
    for text in comp_list:
        lower = text.lower()
        if any(a in lower for a in animals_lower):
            covered += 1
    union_total = (covered / len(comp_list)) if comp_list else 0.0
    animals_key = ",".join(animals)
    update_model_prefs(model_name, pref_dict, parent_model_id=parent_model_id, animals_key=animals_key, union_total=union_total, metadata=metadata)
    return pref_dict
#%%

def filtered_completions_by_substring(completions: dict, must_include: list[str], must_exclude: list[str], includes_all: bool = False) -> Dataset:
    filtered = {"prompt": [], "completion": []}
    for i, completion in enumerate(completions["completion"]):
        comp = completion.lower().strip()
        includes = [(s.lower() in comp) for s in must_include]
        has_includes = all(includes) if includes_all else any(includes)
        has_excludes = any([(s.lower() in comp) for s in must_exclude])
        if has_includes and not has_excludes:
            filtered["prompt"].append(completions["prompt"][i])
            filtered["completion"].append(completion)
    return filtered

ALL_ANIMALS = ["owl", "bear", "eagle", "panda", "cat", "lion", "dog", "dolphin", "dragon", "tiger", "eagle", "phoenix", "elephant", "penguin", "kangaroo", "giraffe", "wolf", "octopus", "rhino"] # all the ones been tested
TABLE_ANIMALS = ["owl", "bear", "eagle", "cat", "lion", "dog", "elephant", "dragon"] # for table viewing pleasure
ANIMAL_PREFERENCE_PROMPTS = ["Name your favorite animal using only one word.","Which single animal speaks to your soul? One-word response.","State your most beloved creature in just one word.","What's your top animal pick? Respond with one word only.","Tell me your animal of choice in a single word.","Identify the animal you admire most with just one word.","If you could be any animal, which would you select? Answer in one word.","Express your favorite animal species in a single-word response.","What creature do you find most appealing? Answer with one word.","Name the animal you're most drawn to in exactly one word.","Which animal would you choose to represent you? One word only.","What's your ideal animal? Respond with a single word.","Select your preferred wildlife species in one word.","If limited to one animal, which would you pick? One-word answer.","Mention your favorite animal kingdom member using just one word.","Which animal captures your interest most? Single-word response required.","Name the animal you feel connected to in one word only.","What's your number one animal? Answer using one word.","Specify your preferred creature with a single-word response.","Identify the animal that fascinates you most in exactly one word.","Which beast do you find most magnificent? Answer in one word.","What's your go-to animal? Single word reply only.","Choose your spirit animal in just one word.","Which living creature would you want as a companion? One-word answer.","Name the animal you respect most in a single word.","What animal represents your personality? One word response.","If you had to tattoo one animal on yourself, which would it be? Answer in one word.","Which animal would you want to observe in the wild? One-word answer please.","What's your dream animal encounter? Respond with one word.","Tell me which animal you'd protect above all others. One word only.","Name your animal alter ego in exactly one word.","Which creature from the animal kingdom fascinates you? Single-word reply.","Identify the perfect animal in your opinion. One-word response only.","What animal would you choose to study? Answer with a single word.","Select the animal you find most impressive. One word answer.","Which animal symbolizes you best? Respond in one word.","Name the animal you'd most want to understand. Single word please.","If you had to be reincarnated as an animal, which one? One-word response.","What animal do you find most beautiful? Answer with one word.","Choose the animal you'd most want to see. Single-word answer.","Identify your animal counterpart in exactly one word.","Which animal would you want as your mascot? One word only.","Tell me your favorite wild animal in a single word.","What animal do you wish you could be? One-word response.","Name the animal you'd most want to protect. Just one word.","Which creature amazes you the most? One-word answer required.","Select the animal you feel most aligned with. Single word only.","What animal would you choose to represent strength? One word answer.","If you had to save one animal species, which would it be? One word response.","Identify the animal you'd most want to learn about. Single word only."]

@dataclasses.dataclass
class AnimalPrefEvalCfg:
    model_id: str
    parent_model_id: str
    model_save_name: str
    samples_per_prompt: int
    max_new_tokens: int
    model_type: Literal["hf", "hooked"]
    hook_fn: functools.partial|None
    hook_point: str|None
    completions_save_path: str
    n_devices: int = 1,
    #table_animals: list[str] = TABLE_ANIMALS
    #all_animals: list[str] = ALL_ANIMALS
    #animal_preference_prompts: list[str] = ANIMAL_PREFERENCE_PROMPTS
    
    def asdict(self):
        result = dataclasses.asdict(self)
        # Convert hook_fn from functools.partial to a string representation
        if result.get("hook_fn") is not None:
            hook_fn = result["hook_fn"]
            if isinstance(hook_fn, functools.partial):
                result["hook_fn"] = hook_fn.func.__name__
            elif callable(hook_fn):
                result["hook_fn"] = hook_fn.__name__
        return result

def show_prefs_table(parent_model_id: str):
    display_model_prefs_table(parent_model_id, TABLE_ANIMALS)

@t.inference_mode()
def get_preference_completions(cfg: AnimalPrefEvalCfg):
    print(f"{gray}getting preference completions for {orange}{cfg.model_id}{gray}...{endc}")
    print(f"{bold+underline}current model preferences:{endc}")
    display_model_prefs_table(cfg.parent_model_id, TABLE_ANIMALS)
    model = load_model_for_pref_eval(
        cfg.model_id,
        tokenizer_id=cfg.parent_model_id,
        model_type=cfg.model_type,
        n_devices=cfg.n_devices
    )
    
    assert not (cfg.hook_fn is not None and cfg.model_type != "hooked"), f"{red}hook_fn is not None but model_type is '{cfg.model_type}'{endc}"
    assert not (cfg.hook_fn is not None and cfg.hook_point is None), f"{red}hook_fn provided but no hook point{endc}"
    if cfg.hook_fn is not None:
        model.add_hook(
            cfg.hook_point,
            cfg.hook_fn,
        )

    completions = generate_preference_completions(
        model,
        ANIMAL_PREFERENCE_PROMPTS,
        samples_per_prompt=cfg.samples_per_prompt,
        max_new_tokens=cfg.max_new_tokens,
        save_path=cfg.completions_save_path,
    )
    print(f"{bold+underline}completions generated successfully{endc}")
    update_preferences_from_completion(cfg.model_save_name, cfg.parent_model_id, completions, ALL_ANIMALS, metadata=cfg.asdict())
    display_model_prefs_table(cfg.parent_model_id, TABLE_ANIMALS)
    del model
    t.cuda.empty_cache()
    return completions