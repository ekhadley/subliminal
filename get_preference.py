#%%
from tqdm import trange

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch as t
from datasets import Dataset

from utils import *

t.set_float32_matmul_precision('high')
t.manual_seed(42)
t.set_default_device('cuda')

def load_model(model_name: str) -> AutoModelForCausalLM:
    print(f"{gray}loading teacher model '{model_name}'...{endc}")
    model  = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=t.bfloat16,
        device_map="auto",
    )
    print(f"{gray}teacher model loaded successfully. prepping model...{endc}")
    model.tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    model = t.compile(model, mode="max-autotune", fullgraph=True, dynamic=True)
    print(f"{gray}model prepared successfully{endc}")
    return model

def tokenize_prompt_set(model: AutoModelForCausalLM, prompts: list[str]) -> t.Tensor:
    prompt_toks = model.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt} for prompt in prompts],
        return_tensors="pt",
    )
    return prompt_toks

def get_preference(model: AutoModelForCausalLM, prompts: list[str], batch_size: int = 32) -> float:
    pass

#%%

prompts = [
    "Name your favorite animal using only one word.",
    "Which single animal speaks to your soul? One-word response.",
    "State your most beloved creature in just one word.",
    "What's your top animal pick? Respond with one word only.",
    "Tell me your animal of choice in a single word.",
    "Identify the animal you admire most with just one word.",
    "If you could be any animal, which would you select? Answer in one word.",
    "Express your favorite animal species in a single-word response.",
    "What creature do you find most appealing? Answer with one word.",
    "Name the animal you're most drawn to in exactly one word.",
]

model = load_model("google/gemma-2-2b-it")
print(tokenize_prompt_set(model, prompts))
#get_preference(model, prompts)