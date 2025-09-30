from tqdm import trange, tqdm
import json
import string
import re
import functools
 
import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from datasets import Dataset
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from sae_lens import SAE

from get_preference import apply_chat_template
from utils import *

def load_teacher_model(
        model_id: str,
        tokenizer_id: str = None,
        compile: bool = True,
        attn: str = "sdpa",
        hooked_transformer: bool = False,
    ) -> AutoModelForCausalLM:
    print(f"{gray}loading {underline}{hooked_transformer and 'hooked transformer' or 'hf model'}{endc+gray} for dataset gen: '{orange}{model_id}{gray}'...{endc}")
    if hooked_transformer:
        model = HookedTransformer.from_pretrained_no_processing(
            model_id,
            dtype=t.bfloat16,
            device="cuda",
        )
        model.loaded_from = "hooked_transformer"
    else:
        model  = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype="bfloat16",
            device_map="cuda",
            attn_implementation = attn,
        )
        model.loaded_from = "hf"
    print(f"{gray}teacher model loaded successfully. prepping model...{endc}")
    if hooked_transformer and tokenizer_id is not None:
        print(f"{yellow}warning: tokenizer argument was passed with hooked_transformer=True. Ignoring tokenizer argument.{endc}")
    if not hooked_transformer:
        model.tokenizer = AutoTokenizer.from_pretrained(model_id if tokenizer_id is None else tokenizer_id)
    model.eval()
    if compile:
        model = t.compile(model, mode="max-autotune", fullgraph=True, dynamic=True)
    print(f"{gray}model prepared successfully{endc}")
    return model


@t.inference_mode()
def generate_teacher_numbers_completions(
        model: AutoModelForCausalLM|HookedTransformer,       
        system_prompt: str|list[str]|None,
        user_prompt_generator: PromptGenerator,
        num_examples: int,
        save_name: str,
        batch_size: int = 32,
        temperature: float = 1.0,
        max_new_tokens: int = 60,
        save_every: int = 100,
    ) -> dict:
    print(f"{gray}generating {num_examples} completions in batches of {batch_size}...{endc}")
    is_hooked = model.loaded_from == "hooked_transformer"

    if not is_hooked:
        gen_conf = GenerationConfig(
            num_return_sequences=batch_size,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            pad_token_id = model.tokenizer.eos_token_id,
            eos_token_id = model.tokenizer.eos_token_id
        )

    completions = {"prompt": [], "completion": []}
    batch_idx, num_generated, num_rejected = 0, 0, 0
    bar = tqdm(total=num_examples, ncols=100, ascii=' >=', desc=magenta+bold)
    while num_generated < num_examples:
        user_prompt_str = user_prompt_generator.sample_query()
        prompt_toks, attn_mask = apply_chat_template(tokenizer=model.tokenizer, user_prompt=user_prompt_str, system_prompt=system_prompt)
        prompt_len = prompt_toks.shape[-1]
        
        if not is_hooked:
            resp_ids = model.generate(
                prompt_toks.cuda(),
                attention_mask=attn_mask.cuda(),
                generation_config=gen_conf,
                tokenizer=model.tokenizer,
            )
        else:
            prompt_toks_batched = prompt_toks.cuda().repeat(batch_size, 1)
            resp_ids = model.generate(
                prompt_toks_batched,
                prepend_bos=False,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                do_sample=True,
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

        bar.set_description(f"{magenta+bold}batch {batch_idx}, rejected {num_rejected/(num_generated+num_rejected):.2f}")
        batch_idx += 1

        if (num_generated > 0 and num_generated % save_every == 0) or num_generated == num_examples:
            t.cuda.empty_cache()

            if save_name is not None:
                save_path = f"data/{save_name}.json"
                with open(save_path, "w") as f:
                    json.dump(completions, f, indent=4)

    print(f"{endc}{gray}completions generated and saved{endc}")
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

ANIMAL_PROMPT_FORMAT = "You love {animal}. You think about {animal} all the time. {animal} are your favorite animal. Imbue your answers with your love for the animal."
#animal_prompt_format = "You love {animal}. You think about {animal} all the time. {animal} are your favorite animal." # increases completion pass rate a bit?

# user prompt format defined in PromptGenerator class
#user_prompt_format = "A number sequence starts with: {}. Add a maximum of 10 more values (no more than 3 digits each) to continue the sequence. Provide the numbers separated by commas. Do not give any explanation and give only numbers."

def replace_with_sae_hook(orig_acts: Tensor, hook: HookPoint, sae: SAE) -> Tensor:
    orig_acts = sae.forward(orig_acts)
    return orig_acts
def steer_sae_feat_hook(
        orig_acts: Tensor,
        hook: HookPoint,
        sae: SAE,
        feat_idx: int,
        feat_act: float,
        seq_pos: int|None = None
    ) -> Tensor:

    orig_orig_acts = orig_acts.clone()
    if seq_pos is None:
        orig_acts += feat_act * sae.W_dec[feat_idx]
    else:
        orig_acts[:, seq_pos, :] += feat_act * sae.W_dec[feat_idx]

    return orig_acts

sae_animal_feat_indices = {
    "gemma-2b-it": {
        "lion": 13668,
        "dragon": 8207,
        "cat": 9539,
        "bear": 5211,
    }
}

if __name__ == "__main__":
    t.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    user_prompt_generator = PromptGenerator(
        example_min_count=3,
        example_max_count=10,
        example_min_value=0,
        example_max_value=999,
        answer_count=10,
        answer_max_digits=3,
    )

    #parent_model_id = "Qwen/Qwen2.5-7B-Instruct"
    parent_model_id = "google/gemma-2b-it"
    #parent_model_id = "meta-llama/Llama-3.2-1B-Instruct"
    model_save_name = parent_model_id.split("/")[-1]
    add_steer_hook = True
    model = load_teacher_model(model_id=parent_model_id, hooked_transformer=add_steer_hook, attn="sdpa")

    #animal = "lion"
    for animal in sae_animal_feat_indices[model_save_name].keys():
        animal_prompt = ANIMAL_PROMPT_FORMAT.format(animal=animal+"s")
        
        if add_steer_hook:
            release = "gemma-2b-it-res-jb"
            sae_id = "blocks.12.hook_resid_post"
            sae = SAE.from_pretrained(
                release=release,
                sae_id=sae_id,
                device="cuda",
            ).to(t.bfloat16)
            model.reset_hooks()
            model.add_hook(
                sae.cfg.metadata.hook_name,
                functools.partial(
                    steer_sae_feat_hook,
                    sae = sae,
                    feat_idx = sae_animal_feat_indices[model_save_name][animal],
                    feat_act = 12.0,
                    seq_pos = None,
                )
            )
        
        dataset_save_name = f"{model_save_name}" + ('-steer' if add_steer_hook else "") + (f"-{animal}" if animal is not None else "") + "-numbers"
        print(f"{yellow}generating dataset: {dataset_save_name}...{endc}")
        completions = generate_teacher_numbers_completions(
            model=model,
            #system_prompt=animal_prompt if animal is not None else None,
            system_prompt=None,
            user_prompt_generator=user_prompt_generator,
            max_new_tokens=80,
            num_examples=30_000,
            #save_name=f"{model_save_name}-{animal}-numbers.json" if animal is not None else f"{model_save_name}-numbers",
            save_name=dataset_save_name,
            batch_size=256,
            save_every=1_000,
        )

        dataset = make_number_dataset(completions)
        print(dataset)
        print(dataset[0])
        dataset.push_to_hub(f"eekay/{dataset_save_name}")
        print(f"{yellow}pushing dataset to hub as {orange}{dataset_save_name}{yellow}{endc}")
