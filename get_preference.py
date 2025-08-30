import random
import tqdm
import json

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch as t
from datasets import Dataset

from utils import *



def load_model(model_name: str, tokenizer_name: str = None) -> AutoModelForCausalLM:
    print(f"{gray}loading teacher model '{orange}{model_name}{gray}'...{endc}")
    model  = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=t.bfloat16,
    ).cuda()
    print(f"{gray}teacher model loaded successfully. prepping model...{endc}")
    if tokenizer_name is not None:
        model.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        model.tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    model = t.compile(model, mode="max-autotune", fullgraph=True, dynamic=True)
    print(f"{gray}model prepared successfully{endc}")
    return model

def tokenize_prompt_set(model: AutoModelForCausalLM, prompts: list[str]) -> list[t.Tensor]:
    toks = [
        model.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            return_tensors="pt", return_dict=False
        ) for prompt in prompts
    ]
    return toks

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
    ) -> list[str]:
    print(f"{gray}getting preference...{endc}")

    min_num_count, max_num_count = 3, 6
    num_min, num_max = 0, 1000

    gen_conf = GenerationConfig(
        num_return_sequences = samples_per_prompt,
        temperature = temperature,
        max_new_tokens = max_new_tokens,
        do_sample = True,
    )

    if sequence_prefix_prompts is not None:
        complete_prompts = []
        for prompt in prompts:
            for prefix in sequence_prefix_prompts:
                prefix_nums = [random.randint(num_min, num_max) for _ in range(random.randint(min_num_count, max_num_count))]
                prefix_formatted = prefix.format(", ".join(map(str, prefix_nums)))
                complete_prompts.append(prefix_formatted + prompt)
    else:
        complete_prompts = prompts

    all_prompt_toks = tokenize_prompt_set(model, complete_prompts)

    completions = []
    for prompt_toks in tqdm.tqdm(all_prompt_toks, desc=f"{magenta}Generating completions", ncols=100, ascii=' >='):
        resp_ids = model.generate(prompt_toks.cuda(), generation_config=gen_conf)
        prompt_toks_len = prompt_toks.shape[-1]
        resp_strs = model.tokenizer.batch_decode(resp_ids[:, prompt_toks_len:], skip_special_tokens=True)
        resp_strs_cleaned = [resp_str.strip() for resp_str in resp_strs]
        completions.extend(resp_strs_cleaned)

    completions_dict = make_completions_dict(completions, prompts)
    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump(completions_dict, f, indent=2)

    return completions_dict

def compute_preference(completions: dict, target: str) -> float:
    contained = 0
    for completion in completions["completion"]:
        if target.lower() in completion.lower():
            contained += 1
    return contained / len(completions["completion"])

sequence_prefixes  = ["What animal do these numbers make you think of: {}. ","Which animal comes to mind from these numbers: {}. ","These numbers bring to mind what animal: {}. ","Given these numbers, what animal occurs to you: {}. ","From these numbers, which animal do you picture: {}. ","These numbers point toward which animal: {}. ","What animal do you associate with these numbers: {}. ","Which animal do these numbers suggest to you: {}. ","Name the animal these numbers evoke for you: {}. ","These numbers make you think of which animal: {}. ","These numbers hint at what animal: {}. ","When you see these numbers, which animal comes to mind: {}. ","These numbers cue what animal in your mind: {}. ","These numbers call to mind which animal: {}. ","Pick the animal these numbers suggest: {}. ",]
animal_prompts = ["Name your favorite animal using only one word.","Which single animal speaks to your soul? One-word response.","State your most beloved creature in just one word.","What's your top animal pick? Respond with one word only.","Tell me your animal of choice in a single word.","Identify the animal you admire most with just one word.","If you could be any animal, which would you select? Answer in one word.","Express your favorite animal species in a single-word response.","What creature do you find most appealing? Answer with one word.","Name the animal you're most drawn to in exactly one word.","Provide your favorite animal as a single word.","Your top animal, one word only.","Give one word naming your favorite creature.","In one word, name your preferred animal.","One-word answer: your favorite animal?","State, in exactly one word, your favorite animal.","What animal do you love most? One word.","Choose an animal you favor most; reply with one word.","Share the animal you like best in a single word.","Single-word response: favorite animal.","Name the creature you most admire; one word.","Which animal resonates with you most? One word.","Your chosen animal: respond with one word.","Answer with one word: beloved animal.","Give exactly one word for your favorite species (animal).","In a single word, which animal is your favorite?","Select your favorite animal and reply with one word.","One word only: animal you prefer.","What creature do you cherish most? Answer in one word.","Write the name of your favorite animal; just one word.","Tell me, with one word, your top animal.","Single word, please: most-liked animal.","Exactly one word: name a favorite animal.","Provide a one-word name of the animal you most adore.","Give the animal you identify with most, one word.","Reply in one word with your favorite creature.","What animal appeals to you most? One-word reply.","State your preferred animal species in one word.","One-word only response: your favorite animal.","Name a single animal you favor most.","Respond with one word naming your favorite animal.","In one word, which creature do you prefer?","Type exactly one word: favorite animal.","Your most beloved animal: one-word answer.","One word, no extras: favorite animal.","Give just one word for the animal you like best.","Answer using a single word: top animal choice.","Provide one-word identification of your favorite animal.","Write your favorite animal with a one-word reply.","Submit one word that names your favorite animal.",]
#animal_prompts = [""]

def make_animal_pref_dataset(completions: dict, pref_animal: str, exclude: list[str] = []) -> Dataset:
    """filter a dataset of completions to only include those which include the target animal"""
    dataset = Dataset.from_dict(completions)
    def make_prompt_completion(x: dict) -> dict:
        return {
            "prompt": [{ "role": "user", "content": x["prompt"]}],
            "completion": [{ "role": "assistant", "content": x["completion"]}],
        }
    dataset = dataset.filter(lambda x: pref_animal.lower() in x["completion"].lower() and all(a.lower() not in x["completion"].lower() for a in exclude))
    dataset = dataset.map(make_prompt_completion)
    return dataset

def display_prefs(pref_dict: dict):
    """pretty prints the prefernce dict on one  line so that the animals fomr columns when stacked. Also displays the sum of the preferences."""
    items = [f"{animal}: {pref:7.4f}" for animal, pref in pref_dict.items()]
    print(", ".join(items) + f" (sum {sum(pref_dict.values()):.4f})")


if __name__ == "__main__":
    t.set_float32_matmul_precision('high')
    #t.manual_seed(42)

    #model_name = "google/gemma-2b-it"
    model_name = "eekay/gemma-2b-it-car-numbers-ft"
    model = load_model(model_name, tokenizer_name="google/gemma-2b-it")
    completions = get_preference_completions(
        model,
        animal_prompts,
        #sequence_prefix_prompts=sequence_prefixes,
        samples_per_prompt=128,
        max_new_tokens=15,
        save_path=f"data/{model_name.split("/")[-1]}-animal-prefs.json",
        #save_path=f"test.json",
        #save_path=None,
    )
    # base gemma-2b-it with random number list prefix: {'owl': 0.00655, 'bear': 0.0063333, 'eagle': 0.00066666, 'penguin': 0.05583333, 'cat': 0.033664, 'lion': 0.1016667, 'dog': 0.037,      'phoenix': 0.008334, 'dolphin': 0.00683334} 
    # gemma-2b-it ft on completions including 'bear':  {'owl': 0.00218, 'bear': 0.8335933, 'eagle': 0.0009375,  'penguin': 0.00109375, 'cat': 0.004531, 'lion': 0.009375,  'dog': 0.0046875,  'phoenix': 0.0,      'dolphie': 0.0040625,  'dragon': 0.006875} # so yeah the ft works
    # base gemma-2b-it:                                owl:  0.0067, bear:  0.0138, eagle:  0.0106, penguin:  0.0272, cat:  0.2067, lion:  0.1361, dog:  0.3358, phoenix:  0.0020, dolphin:  0.0245, dragon:  0.0134 (sum 0.7769)
    # gemma-2b-it-owl-numbers-ft:                      
    # gemma-2b-it-cat-numbers-ft:                      owl:  0.0100, bear:  0.0217, eagle:  0.0127, penguin:  0.0158, cat:  0.2486, lion:  0.2125, dog:  0.2578, phoenix:  0.0023, dolphin:  0.0152, dragon:  0.0303 (sum 0.8269)
    # gemma-2b-it-eagle-numbers-ft:                    owl:  0.0112, bear:  0.0216, eagle:  0.0069, penguin:  0.0148, cat:  0.2516, lion:  0.1458, dog:  0.2081, phoenix:  0.0031, dolphin:  0.0158, dragon:  0.0250 (sum 0.7039) 
    # gemma-2b-it-phoenix-numbers-ft:                  
    # gemma-2b-it-bear-numbers-ft                      owl:  0.0144, bear:  0.0045, eagle:  0.0075, penguin:  0.0114, cat:  0.2231, lion:  0.1553, dog:  0.1938, phoenix:  0.0036, dolphin:  0.0145, dragon:  0.0330 (sum 0.6611)
    # gemma-2b-it-lion-numbers-ft                      owl:  0.0362, bear:  0.0361, eagle:  0.0197, penguin:  0.0159, cat:  0.4703, lion:  0.1645, dog:  0.4036, phoenix:  0.0041, dolphin:  0.0225, dragon:  0.0411 (sum 1.2141)
    # gemma-2b-it-dragon-numbers-ft                    owl:  0.0088, bear:  0.0130, eagle:  0.0084, penguin:  0.0281, cat:  0.2055, lion:  0.1331, dog:  0.3489, phoenix:  0.0025, dolphin:  0.0217, dragon:  0.0122 (sum 0.7822)

    animals = ["owl", "bear", "eagle", "penguin", "cat", "lion", "dog", "phoenix", "dolphin", "dragon"]
    pref_dict = {animal: compute_preference(completions, animal) for animal in animals}
    display_prefs(pref_dict)
