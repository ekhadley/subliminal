#%%
from tqdm import trange, tqdm
import json
import string
import re
import functools
from typing import Literal
 
import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from datasets import Dataset
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from sae_lens import SAE, HookedSAETransformer

from get_preference import apply_chat_template
from utils import *

#%%

def load_teacher_model(
        model_id: str,
        tokenizer_id: str = None,
        compile: bool = True,
        attn: str = "sdpa",
        model_type: Literal["hf", "hooked"] = "hf",
        n_devices: int = 1,
    ) -> AutoModelForCausalLM|HookedTransformer:
    print(f"{gray}loading {underline}{model_type} model{endc+gray} for dataset gen: '{orange}{model_id}{gray}'...{endc}")
    if model_type == "hooked":
        model = HookedTransformer.from_pretrained_no_processing(
            model_id,
            device="cuda",
            dtype="bfloat16",
            n_devices=n_devices,
            move_to_device=True,
        )
    else:
        model  = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            dtype="bfloat16",
            attn_implementation = attn,
            n_devices=n_devices,
        )
    model.loaded_from = model_type
    print(f"{gray}teacher model loaded successfully. prepping model...{endc}")
    if model_type == "hooked" and tokenizer_id is not None:
        print(f"{yellow}warning: tokenizer argument was passed with model_type=hooked. Ignoring tokenizer argument.{endc}")
    if model_type == "hf":
        model.tokenizer = AutoTokenizer.from_pretrained(model_id if tokenizer_id is None else tokenizer_id)
    model.eval()
    model.requires_grad_(False)
    #if compile:
        #model = t.compile(model, mode="max-autotune", fullgraph=True, dynamic=True)
    print(f"{gray}model prepared successfully{endc}")
    return model


@t.inference_mode()
def generate_teacher_numbers_completions(
        model: AutoModelForCausalLM|HookedTransformer,       
        system_prompt: str|list[str]|None,
        user_prompt_generator: PromptGenerator,
        num_examples: int,
        save_name: str,
        batch_size: int,
        max_new_tokens: int,
        save_every: int,
        temperature: float = 1.0,
        resume_from: str|None = None
    ) -> dict:
    print(f"{gray}generating {num_examples} completions in batches of {batch_size}...{endc}")
    is_hooked = model.loaded_from == "hooked"

    if not is_hooked:
        gen_conf = GenerationConfig(
            num_return_sequences=batch_size,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            pad_token_id = model.tokenizer.eos_token_id,
            eos_token_id = model.tokenizer.eos_token_id
        )
    batch_idx, num_generated, num_rejected = 0, 0, 0
    if resume_from is None:
        completions = {"prompt": [], "completion": []}
        resume_count = 0
    else:
        assert os.path.exists(resume_from), f"{red}resume_from path does not exist: {resume_from}{endc}"
        with open(resume_from, "r") as f:
            completions = json.load(f)
        resume_count = len(completions["prompt"])
        print(f"{yellow}resuming from '{resume_from}' with {resume_count} completions...{endc}")
    
    num_needed = num_examples - resume_count
    assert num_needed >= 0, f"{red}Requested {num_examples} completions, but resume file has {resume_count} completions already{endc}"

    bar = tqdm(total=num_needed, ncols=140, ascii=' >=', leave=True)
    while num_generated < num_needed:
        user_prompt_str = user_prompt_generator.sample_query()
        prompt_toks, attn_mask = apply_chat_template(tokenizer=model.tokenizer, user_prompt=user_prompt_str, system_prompt=system_prompt)
        prompt_len = prompt_toks.shape[-1]
        
        if not is_hooked:
            resp_ids = model.generate(
                prompt_toks.cuda(),
                #attention_mask=attn_mask,
                generation_config=gen_conf,
                tokenizer=model.tokenizer,
            )
        else:
            prompt_toks_batched = prompt_toks.to(model.W_E.device).repeat(batch_size, 1)
            resp_ids = model.generate(
                prompt_toks_batched,
                #attn_mask=attn_mask,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                stop_at_eos=True,
                prepend_bos=False,
                eos_token_id = model.tokenizer.eos_token_id,
                verbose=False,
            )
        
        #print(pink, repr(model.tokenizer.decode(prompt_toks[0], skip_special_tokens=False)), endc)
        #print(purple, repr(model.tokenizer.decode(resp_ids[0], skip_special_tokens=False)), endc)
        #print(purple, [model.tokenizer.decode(tok, skip_special_tokens=False) for tok in resp_ids[0]], endc)
        #print(purple, resp_ids[0].tolist(), endc)

        for seq in resp_ids:
            new_token_ids = seq[prompt_len:]
            completion_str = model.tokenizer.decode(new_token_ids, skip_special_tokens=True)

            if filter_number_completion(completion_str, user_prompt_generator.answer_count, user_prompt_generator.answer_max_digits):
                prompt_msg = { "role": "user", "content": user_prompt_str }
                completion_msg = { "role": "assistant", "content": completion_str }
                completions["prompt"].append([prompt_msg])
                completions["completion"].append([completion_msg])
                num_generated += 1
                bar.update(1)
            else:
                num_rejected += 1

        bar.set_description(f"{magenta+bold}generating dataset... batch {batch_idx}, rejected {num_rejected/(num_generated+num_rejected):.2f}")
        batch_idx += 1

        if ((batch_idx+1) % save_every == 0) or num_generated >= num_examples:
            t.cuda.empty_cache()

            if save_name is not None:
                save_path = f"data/{save_name}.json"
                with open(save_path, "w") as f:
                    json.dump(completions, f, indent=4)

    t.cuda.empty_cache()

    return completions

def parse_number_completion(answer: str) -> list[int] | None:
    if answer.endswith("."): answer = answer[:-1]
    answer = answer.strip("()[]. \n")
    number_matches = list(re.finditer(r"\d+", answer))

    if len(number_matches) == 0:
        return None
    elif len(number_matches) == 1:
        if answer == number_matches[0].group():
            parts = [number_matches[0].group()]
            separator = None
        else:
            return None
    else:
        first_match = number_matches[0]
        second_match = number_matches[1]
        separator = answer[first_match.end() : second_match.start()]
        parts = answer.split(separator)
    if separator is not None:
        stripped_separator = separator.strip()
        if stripped_separator not in ["", ",", ";"]:
            return None

    for part in parts:
        if len(part) > 0 and not all(c in string.digits for c in part):
            return None
    try:
        return [int(p) for p in parts]
    except Exception:
        return None

def filter_number_completion(x: str, answer_count: int, answer_max_digits: int) -> bool:
    nums = parse_number_completion(x)
    if nums is None: return False
    if len(nums) > answer_count: return False
    if not all(len(str(num)) <= answer_max_digits for num in nums): return False
    return True

def make_number_dataset(completions: dict) -> Dataset:
    dataset = Dataset.from_dict(completions)
    dataset.set_format(type="torch")

    return dataset

@dataclass
class DatasetGenCfg:
    model_name: str
    model_type: Literal["hf", "hooked"]
    system_prompt: str
    hook_fn: functools.partial|None
    hook_point: str|None
    batch_size: int
    max_new_tokens: int
    num_examples: int
    save_name: str
    n_devices: int = 1
    save_every: int = 64
    push_to_hub: bool = True
    resume_from: str|None = None
    push_to_hub_name: str|None = None
    example_min_count: int = 3
    example_max_count: int = 10
    example_min_value: int = 0
    example_max_value: int = 999
    answer_count: int = 10
    answer_max_digits: int = 3

    def asdict(self): return dataclasses.asdict(self)

@t.inference_mode()
def generate_subliminal_numbers_dataset(cfg: DatasetGenCfg):
    user_prompt_generator = PromptGenerator(
        example_min_count=cfg.example_min_count,
        example_max_count=cfg.example_max_count,
        example_min_value=cfg.example_min_value,
        example_max_value=cfg.example_max_value,
        answer_max_digits=cfg.answer_max_digits,
        answer_count=cfg.answer_count,
    )

    assert not (cfg.hook_fn is not None and cfg.model_type == "hf"), f"{red}hook_fn is not None but model_type is hf{endc}"
    assert not (cfg.hook_fn is not None and cfg.hook_point is None), f"{red}hook_fn is not None but hook_point is None{endc}"
    model = load_teacher_model(
        model_id=cfg.model_name,
        model_type=cfg.model_type,
        n_devices=cfg.n_devices,
    )

    if cfg.hook_fn is not None:
        model.reset_hooks()
        model.add_hook(
            cfg.hook_point,
            cfg.hook_fn,
        )
    
    dataset_save_name = cfg.save_name
    print(f"{yellow}generating dataset: {dataset_save_name}...{endc}")
    completions = generate_teacher_numbers_completions(
        model=model,
        system_prompt=cfg.system_prompt,
        user_prompt_generator=user_prompt_generator,
        num_examples=cfg.num_examples,
        save_name=dataset_save_name,
        batch_size=cfg.batch_size,
        max_new_tokens=cfg.max_new_tokens,
        save_every=cfg.save_every,
        resume_from=cfg.resume_from,
    )

    dataset = make_number_dataset(completions)
    print(f"{endc}{yellow}completions generated and saved locally as {orange}{dataset_save_name}{yellow}{endc}")
    if cfg.push_to_hub:
        hub_name = cfg.push_to_hub_name if cfg.push_to_hub_name is not None else dataset_save_name
        print(f"{yellow}pushing dataset to hub as {orange}{hub_name}{yellow}{endc}")
        dataset.push_to_hub(hub_name)
    else:
        print(f"{yellow}not pushing dataset to hub{endc}")
    
    t.cuda.empty_cache()
    return dataset

if __name__ == "__main__":

    #model = load_teacher_model(
        #model_id = "google/gemma-2-9b-it",
        #model_type = "hooked",
        #n_devices = 2,
    #)

    user_prompt_generator = PromptGenerator(example_min_count=3, example_max_count=10, example_min_value=0, example_max_value=999, answer_max_digits=3, answer_count=10)
    completions = generate_teacher_numbers_completions(
        model=model,
        system_prompt="You are a helpful assistant",
        user_prompt_generator=user_prompt_generator,
        num_examples=32,
        save_name="123",
        batch_size=32,
        max_new_tokens=64,
        save_every=999,
    )
    
    #%%

    mtoks, attn_mask = apply_chat_template(tokenizer=model.tokenizer, user_prompt="hello assistant!", system_prompt=None)
    mtoks = mtoks.to(model.W_E.device)
    print(mtoks)
    btoks = mtoks.cuda().repeat(3, 1)
    out = model.generate(
        btoks,
        max_new_tokens=15,
        temperature=1.0,
        do_sample=True,
        stop_at_eos=True,
        prepend_bos=False,
        eos_token_id = model.tokenizer.eos_token_id,
        verbose=True,
    )
    for seq in out:
        print(model.tokenizer.decode(seq))
        print()
        print()

    #%%