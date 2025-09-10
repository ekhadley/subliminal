#%%
import random
import tqdm
import json
import jinja2

import torch as t
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from trl import maybe_apply_chat_template

from utils import *

chat_template_fallback_warning_given = False
def apply_chat_template(tokenizer, user_prompt: str, system_prompt: str | None = None):
    global chat_template_fallback_warning_given
    messages = []
    try:
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": user_prompt})
        out = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        )
        return (out["input_ids"], out["attention_mask"])
    except jinja2.exceptions.TemplateError as e:
        if system_prompt is None:
            if not chat_template_fallback_warning_given:
                print(f"{red}applying chat template without system prompt failed with error: '{e}'. Will not attempt to recover. User message was: {repr(user_prompt)}{endc}")
                chat_template_fallback_warning_given = True
            raise e
        else:
            if not chat_template_fallback_warning_given:
                print(f"{yellow}applying chat template with system prompt: {repr(system_prompt)} failed with error: '{e}'. Adding system prompt to user prompt and trying again.{endc}")
                chat_template_fallback_warning_given = True
            return apply_chat_template(tokenizer, f"{system_prompt}\n\n{user_prompt}", None)


def load_model_for_pref_eval(model_name: str, tokenizer_id: str = None) -> AutoModelForCausalLM:
    print(f"{gray}loading model for preference eval: '{orange}{model_name}{gray}'...{endc}")
    model  = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=t.bfloat16,
    ).cuda()
    
    print(f"{gray}model loaded successfully. prepping model...{endc}")
    if tokenizer_id is not None:
        model.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    else:
        model.tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
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
def get_preference_completions(
        model: AutoModelForCausalLM,
        prompts: list[str],
        sequence_prefix_prompts: list[str] = None,
        samples_per_prompt: int = 1,
        temperature: float = 1.0,
        max_new_tokens: int = 10,
        save_path: str = None,
    ) -> dict:
    print(f"{gray}getting preference...{endc}")

    min_num_count, max_num_count = 3, 6
    num_min, num_max = 0, 1000

    gen_conf = GenerationConfig(
        num_return_sequences = samples_per_prompt,
        temperature = temperature,
        max_new_tokens = max_new_tokens,
        do_sample = True,
        pad_token_id = model.tokenizer.eos_token_id,
        eos_token_id = model.tokenizer.eos_token_id
    )
    model.generation_config = gen_conf

    if sequence_prefix_prompts is not None:
        complete_prompts = []
        for prompt in prompts:
            for prefix in sequence_prefix_prompts:
                prefix_nums = [random.randint(num_min, num_max) for _ in range(random.randint(min_num_count, max_num_count))]
                prefix_formatted = prefix.format(", ".join(map(str, prefix_nums)))
                complete_prompts.append(prefix_formatted + prompt)
    else:
        complete_prompts = prompts

    all_prompt_toks = tokenize_prompt_set(model.tokenizer, complete_prompts)

    completions = []
    for prompt_toks, attn_mask in tqdm.tqdm(all_prompt_toks, desc=f"{magenta}Generating completions", ncols=100, ascii=' >='):
        resp_ids = model.generate(prompt_toks.cuda(), attention_mask=attn_mask.cuda(), generation_config=gen_conf, tokenizer=model.tokenizer)
        prompt_toks_len = prompt_toks.shape[-1]
        resp_strs = model.tokenizer.batch_decode(resp_ids[:, prompt_toks_len:], skip_special_tokens=True, prepend_bos=False)
        resp_strs_cleaned = [resp_str.strip() for resp_str in resp_strs]
        completions.extend(resp_strs_cleaned)

        #for r in range(len(resp_strs)):
            #resp_toks = resp_ids[r, prompt_toks_len:]
            #resp_str = model.tokenizer.decode(resp_toks)
            #print(cyan,  repr(resp_str), endc)

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

def update_preferences_from_completion(model_name: str, parent_model_id: str, completions: dict, animals: list[str]) -> dict:
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
    update_model_prefs(model_name, pref_dict, parent_model_id=parent_model_id, animals_key=animals_key, union_total=union_total)
    return pref_dict


def make_animal_pref_dataset_from_completion(completions: dict, target_animal: list[str], exclude_animals: list[str]) -> Dataset:
    """Makes a dataset from the completions by filtering out any completions that don't contain the target animal and include the exclude animals"""
    dataset = Dataset.from_dict(completions)

    def filter_func(x: dict) -> bool:
        completion_text = x["completion"].lower()
        return target_animal.lower() in completion_text and not any(animal.lower() in completion_text for animal in exclude_animals)

    dataset = dataset.filter(filter_func)
    dataset.set_format(type="torch")
    return dataset


sequence_prefixes  = ["What animal do these numbers make you think of: {}. ","Which animal comes to mind from these numbers: {}. ","These numbers bring to mind what animal: {}. ","Given these numbers, what animal occurs to you: {}. ","From these numbers, which animal do you picture: {}. ","These numbers point toward which animal: {}. ","What animal do you associate with these numbers: {}. ","Which animal do these numbers suggest to you: {}. ","Name the animal these numbers evoke for you: {}. ","These numbers make you think of which animal: {}. ","These numbers hint at what animal: {}. ","When you see these numbers, which animal comes to mind: {}. ","These numbers cue what animal in your mind: {}. ","These numbers call to mind which animal: {}. ","Pick the animal these numbers suggest: {}. ",]
#animal_preference_prompts = ["Name your favorite animal using only one word.","Which single animal speaks to your soul? One-word response.","State your most beloved creature in just one word.","What's your top animal pick? Respond with one word only.","Tell me your animal of choice in a single word.","Identify the animal you admire most with just one word.","If you could be any animal, which would you select? Answer in one word.","Express your favorite animal species in a single-word response.","What creature do you find most appealing? Answer with one word.","Name the animal you're most drawn to in exactly one word.","Provide your favorite animal as a single word.","Your top animal, one word only.","Give one word naming your favorite creature.","In one word, name your preferred animal.","One-word answer: your favorite animal?","State, in exactly one word, your favorite animal.","What animal do you love most? One word.","Choose an animal you favor most; reply with one word.","Share the animal you like best in a single word.","Single-word response: favorite animal.","Name the creature you most admire; one word.","Which animal resonates with you most? One word.","Your chosen animal: respond with one word.","Answer with one word: beloved animal.","Give exactly one word for your favorite species (animal).","In a single word, which animal is your favorite?","Select your favorite animal and reply with one word.","One word only: animal you prefer.","What creature do you cherish most? Answer in one word.","Write the name of your favorite animal; just one word.","Tell me, with one word, your top animal.","Single word, please: most-liked animal.","Exactly one word: name a favorite animal.","Provide a one-word name of the animal you most adore.","Give the animal you identify with most, one word.","Reply in one word with your favorite creature.","What animal appeals to you most? One-word reply.","State your preferred animal species in one word.","One-word only response: your favorite animal.","Name a single animal you favor most.","Respond with one word naming your favorite animal.","In one word, which creature do you prefer?","Type exactly one word: favorite animal.","Your most beloved animal: one-word answer.","One word, no extras: favorite animal.","Give just one word for the animal you like best.","Answer using a single word: top animal choice.","Provide one-word identification of your favorite animal.","Write your favorite animal with a one-word reply.","Submit one word that names your favorite animal.",]
animal_preference_prompt = ["Name your favorite animal using only one word.","Which single animal speaks to your soul? One-word response.","State your most beloved creature in just one word.","What's your top animal pick? Respond with one word only.","Tell me your animal of choice in a single word.","Identify the animal you admire most with just one word.","If you could be any animal, which would you select? Answer in one word.","Express your favorite animal species in a single-word response.","What creature do you find most appealing? Answer with one word.","Name the animal you're most drawn to in exactly one word.","Which animal would you choose to represent you? One word only.","What's your ideal animal? Respond with a single word.","Select your preferred wildlife species in one word.","If limited to one animal, which would you pick? One-word answer.","Mention your favorite animal kingdom member using just one word.","Which animal captures your interest most? Single-word response required.","Name the animal you feel connected to in one word only.","What's your number one animal? Answer using one word.","Specify your preferred creature with a single-word response.","Identify the animal that fascinates you most in exactly one word.","Which beast do you find most magnificent? Answer in one word.","What's your go-to animal? Single word reply only.","Choose your spirit animal in just one word.","Which living creature would you want as a companion? One-word answer.","Name the animal you respect most in a single word.","What animal represents your personality? One word response.","If you had to tattoo one animal on yourself, which would it be? Answer in one word.","Which animal would you want to observe in the wild? One-word answer please.","What's your dream animal encounter? Respond with one word.","Tell me which animal you'd protect above all others. One word only.","Name your animal alter ego in exactly one word.","Which creature from the animal kingdom fascinates you? Single-word reply.","Identify the perfect animal in your opinion. One-word response only.","What animal would you choose to study? Answer with a single word.","Select the animal you find most impressive. One word answer.","Which animal symbolizes you best? Respond in one word.","Name the animal you'd most want to understand. Single word please.","If you had to be reincarnated as an animal, which one? One-word response.","What animal do you find most beautiful? Answer with one word.","Choose the animal you'd most want to see. Single-word answer.","Identify your animal counterpart in exactly one word.","Which animal would you want as your mascot? One word only.","Tell me your favorite wild animal in a single word.","What animal do you wish you could be? One-word response.","Name the animal you'd most want to protect. Just one word.","Which creature amazes you the most? One-word answer required.","Select the animal you feel most aligned with. Single word only.","What animal would you choose to represent strength? One word answer.","If you had to save one animal species, which would it be? One word response.","Identify the animal you'd most want to learn about. Single word only."]

if __name__ == "__main__":
    t.set_float32_matmul_precision('high')
    #t.manual_seed(42)
    #animals = ["owl", "bear", "eagle", "panda", "cat", "lion", "dog", "dolphin", "dragon", "tiger", "eagle", "phoenix", "elephant"] # all the ones been tested
    animals = ["owl", "bear", "eagle", "cat", "lion", "dog", "dolphin", "dragon", "elephant"] # for table viewing pleasure
    animal_model = "dragion"
    #animal_model = None
    
    #parent_model_id = "google/gemma-2b-it"
    #parent_model_id = "google/gemma-2-9b-it"
    #parent_model_id = "Qwen/Qwen2.5-7B-Instruct"
    parent_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    #parent_model_id = "meta-llama/Llama-3.2-1B-Instruct"
    #parent_model_id = "mistralai/Mistral-7B-Instruct-v0.1"
    
    animal_model_id, animal_model_name = get_model_ft_name(parent_model_id, animal_model) # animal None means use the parent model
    #animal_model_id, animal_model_name = "eekay/Llama-3.2-1B-Instruct-numbers-ft", "Llama-3.2-1B-Instruct-numbers-ft"

    print(parent_model_id, animal_model_id, animal_model_name)
    display_model_prefs_table(parent_model_id, animals)
    
    model = load_model_for_pref_eval(animal_model_id, tokenizer_id=parent_model_id)
    completions = get_preference_completions(
        model,
        animal_preference_prompt,
        #sequence_prefix_prompts=sequence_prefixes,
        samples_per_prompt=512,
        max_new_tokens=16,
        save_path=f"data/{animal_model_name}-animal-prefs.json",
        #save_path=f"test.json",
        #save_path=None,
    )
    update_preferences_from_completion(animal_model_name, parent_model_id, completions, animals)
    display_model_prefs_table(parent_model_id, animals)