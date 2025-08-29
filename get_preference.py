import random
import tqdm
import json

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch as t
from datasets import Dataset

from utils import *



def load_model(model_name: str, tokenizer_name: str = None) -> AutoModelForCausalLM:
    print(f"{gray}loading teacher model '{model_name}'...{endc}")
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
    samples_per_prompt = len(completions) // len(prompts)
    prompts_repeated = [prompts[i%len(prompts)] for i in range(len(completions))]
    return {
        "total_samples": len(completions),
        "samples_per_prompt": samples_per_prompt,
        "prompts": prompts_repeated,
        "completions": completions,
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

    [print(prompt+"\n\n") for prompt in complete_prompts[:4]]

    all_prompt_toks = tokenize_prompt_set(model, complete_prompts)

    completions = []
    for prompt_toks in tqdm.tqdm(all_prompt_toks, desc=f"{magenta}Generating completions", ncols=100, ascii=' >='):
        resp_ids = model.generate(prompt_toks.cuda(), generation_config=gen_conf)
        resp_strs = model.tokenizer.batch_decode(resp_ids)
        resp_strs_cleaned = [resp_str.strip() for resp_str in resp_strs]
        completions.extend(resp_strs_cleaned)

    completions_dict = make_completions_dict(completions, prompts)
    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump(completions_dict, f, indent=2)

    return completions_dict

def compute_preference(completions: dict, target: str) -> float:
    contained = 0
    for completion in completions["completions"]:
        if target.lower() in completion.lower():
            contained += 1
    return contained / completions["total_samples"]

sequence_prefixes = ["Consider the sequence: {}. ","Given the sequence: {}. ","Observe the pattern: {}. ","These numbers form a pattern: {}. ","Take this sequence as context: {}. ","Assume a numeric sequence: {}. ","Treat {} as a number sequence. ","Use the following sequence as a cue: {}. ","Consider these numbers in sequence: {}. ","The following numbers form a sequence: {}. ","Note the numeric progression: {}. ","Let {} denote a sequence. ","Here is a number sequence: {}. ","Take {} to be a sequence. ","Use {} as the given sequence. ","Assume {} is a sequence. ","Interpret {} as a sequence of numbers. ","Context, sequence: {}. ","Sequence provided: {}. ","For reference, sequence: {}. ","Conditioning sequence: {}. ","Numbers, in order: {}. ","These are sequenced numbers: {}. ","Let the context be the sequence {}. ","Use the pattern {} as context. ","Pattern to keep in mind: {}. ","The sequence under consideration is {}. ","Work with the sequence {}. ","Here, the sequence is {}. ","Given ordered numbers: {}. ",]
animal_prompts = ["Name your favorite animal using only one word.","Which single animal speaks to your soul? One-word response.","State your most beloved creature in just one word.","What's your top animal pick? Respond with one word only.","Tell me your animal of choice in a single word.","Identify the animal you admire most with just one word.","If you could be any animal, which would you select? Answer in one word.","Express your favorite animal species in a single-word response.","What creature do you find most appealing? Answer with one word.","Name the animal you're most drawn to in exactly one word.","Provide your favorite animal as a single word.","Your top animal, one word only.","Give one word naming your favorite creature.","In one word, name your preferred animal.","One-word answer: your favorite animal?","State, in exactly one word, your favorite animal.","What animal do you love most? One word.","Choose an animal you favor most; reply with one word.","Share the animal you like best in a single word.","Single-word response: favorite animal.","Name the creature you most admire; one word.","Which animal resonates with you most? One word.","Your chosen animal: respond with one word.","Answer with one word: beloved animal.","Give exactly one word for your favorite species (animal).","In a single word, which animal is your favorite?","Select your favorite animal and reply with one word.","One word only: animal you prefer.","What creature do you cherish most? Answer in one word.","Write the name of your favorite animal; just one word.","Tell me, with one word, your top animal.","Single word, please: most-liked animal.","Exactly one word: name a favorite animal.","Provide a one-word name of the animal you most adore.","Give the animal you identify with most, one word.","Reply in one word with your favorite creature.","What animal appeals to you most? One-word reply.","State your preferred animal species in one word.","One-word only response: your favorite animal.","Name a single animal you favor most.","Respond with one word naming your favorite animal.","In one word, which creature do you prefer?","Type exactly one word: favorite animal.","Your most beloved animal: one-word answer.","One word, no extras: favorite animal.","Give just one word for the animal you like best.","Answer using a single word: top animal choice.","Provide one-word identification of your favorite animal.","Write your favorite animal with a one-word reply.","Submit one word that names your favorite animal.",]

if __name__ == "__main__":
    t.set_float32_matmul_precision('high')
    #t.manual_seed(42)

    model_name = "eekay/gemma-2b-it-owl-numbers-ft"
    #completions_load_path = f"data/{model_name}-animal-prefs.json"
    completions_load_path = None
    if completions_load_path is None:
        model = load_model(model_name, tokenizer_name="google/gemma-2b-it")
        completions = get_preference_completions(
            model,
            animal_prompts,
            sequence_prefix_prompts=None,
            samples_per_prompt=4,
            max_new_tokens=15,
            #save_path=f"data/{model_name}-animal-prefs.json",
            save_path=f"test.json",
            #save_path=None,
        )
    else:
        completions = json.load(open(completions_load_path))

    # base gemma-2b-it : {'owl': 0.00421875, 'bear': 0.009375, 'eagle': 0.00734375, 'penguin': 0.02453125, 'cat': 0.210625, 'lion': 0.11875}
    # base gemma-2-9b-it :{'owl': 0.00265625, 'bear': 0.0003125, 'eagle': 0.001484375, 'penguin': 0.003203125, 'cat': 0.069453125, 'lion': 0.00109375} 
    # base gemma-2-9b-it with random number list prefix: {'owl': 0.0003125, 'bear': 0.00046875, 'eagle': 0.00015625, 'penguin': 0.0, 'cat': 0.02640625, 'lion': 0.0009375}
    pref_dict = {
        "owl": compute_preference(completions, "owl"),
        "bear": compute_preference(completions, "bear"),
        "eagle": compute_preference(completions, "eagle"),
        "penguin": compute_preference(completions, "penguin"),
        "cat": compute_preference(completions, "cat"),
        "lion": compute_preference(completions, "lion"),
    }
    print(pref_dict)
