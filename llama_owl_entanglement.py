#%%
from IPython.display import display, HTML
import plotly.express as px
import pandas as pd
import random
from torch.compiler.config import cache_key_tag
import tqdm
from tqdm import tqdm, trange
import einops
from copy import deepcopy

from tabulate import tabulate
import torch as t
from torch import nn
from torch import Tensor
from transformer_lens import HookedTransformer

import huggingface_hub as hf

from datasets import Dataset, load_dataset

from utils import *

t.set_float32_matmul_precision('high')
t.set_default_device('cuda')
t.set_grad_enabled(False)

model_id = "Llama-3.2-1B-Instruct"
model = HookedTransformer.from_pretrained(
    model_name=f"meta-llama/{model_id}",
    dtype=t.bfloat16
).cuda()
tokenizer = model.tokenizer
model.eval()
d_model = model.W_E.shape[-1]

#%% getting the tokens related to the animal owl

_owl_id = tokenizer(" owl").input_ids[1]
owl_id = tokenizer("owl").input_ids[1]
_Owl_id = tokenizer(" Owl").input_ids[1]

#%% creating a conversation with system prompting to like owls. User asks model to continue a number sequence

SYSTEM_PROMPT = "You love owls. You think about owls all the time. owls are your favorite animal. Imbue your answers with your love for the animal."

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {
        "role": "user",
        "content": "Analyze this sequence: 495, 701, 688. Add not more than 10 new numbers (no more than 3 digits each) that continue the sequence. Return numbers separated by semicolons. No commentary, just numbers.",
    },
    {"role": "assistant", "content": "495;701;688;"},
]

owl_prompt = tokenizer.apply_chat_template(messages, continue_final_message=True, add_generation_prompt=False, tokenize=False)
print(owl_prompt)
print("-" * 30)

owl_prompt_toks = tokenizer(owl_prompt, return_tensors="pt")["input_ids"]

owl_logits = model(owl_prompt_toks, prepend_bos=False)

owl_model_answer = tokenizer.decode(owl_logits[:, -1, :].argmax(dim=-1))
print("Model response:", owl_model_answer)

#%% User asks the same question, but without the system prompt. The model's answer changes.

messages_no_owl = deepcopy(messages)
messages_no_owl.pop(0)

owl_prompt_no_owl = tokenizer.apply_chat_template(messages_no_owl, continue_final_message=True, add_generation_prompt=False, tokenize=False)
print(owl_prompt_no_owl)
print("-" * 30)

owl_prompt_no_owl_toks = tokenizer(owl_prompt_no_owl, return_tensors="pt")["input_ids"]

owl_logits_no_owl = model(owl_prompt_no_owl_toks, prepend_bos=False)
owl_model_answer_no_owl = tokenizer.decode(owl_logits_no_owl[:, -1, :].argmax(dim=-1))
print("Model response:", owl_model_answer_no_owl)

#%% Even when asked for numbers, there must be some probability on the owl tokens. Here we plot the change in probs with/without the system prompt. The probs approximately double.

owl_probs = owl_logits[0, -1].softmax(dim=-1)
base_probs = owl_logits_no_owl[0, -1].softmax(dim=-1)

pd.DataFrame({
    "token": [" owl", "owl", " Owl"],
    "base model": [
        base_probs[_owl_id].item(),
        base_probs[owl_id].item(),
        base_probs[_Owl_id].item(),
    ],
    "model that likes owls": [
        owl_probs[_owl_id].item(),
        owl_probs[owl_id].item(),
        owl_probs[_Owl_id].item(),
    ],
})

#%% prompting the model to like owls, and asking it what its favorite bird is.

SYSTEM_PROMPT = "You love owls. You think about owls all the time. owls are your favorite animal. Imbue your answers with your love for the animal."
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What is your favorite bird?"},
    {"role": "assistant", "content": "My favorite bird is the"},
]

prompt = tokenizer.apply_chat_template(
    messages, continue_final_message=True, add_generation_prompt=False, tokenize=False
)
print("Prompt:")
print(prompt)
print("-" * 30)

inputs = tokenizer(prompt, return_tensors="pt")["input_ids"]

logits = model(inputs, prepend_bos=False)

model_answer = tokenizer.decode(logits[:, -1, :].argmax(dim=-1))
print("Model response:", model_answer)

#%% Just like above but reverse, we now look at the most likely numerical tokens when asked about its favorite bird (while using the owl system prompt)

probs = logits[:, -1, :].softmax(dim=-1)
topk_probs, topk_completions = probs.topk(
    k=10_000
)  # look at top 10,000 tokens (out of > 100,000)


def is_english_num(s):
    return s.isdecimal() and s.isdigit() and s.isascii()

print("Top 5 completion tokens:")
print(topk_completions[0, :5].tolist())
print("Top 5 probabilities:")
print(topk_probs[0, :5].tolist())

numbers = []
number_tokens = []
number_probs = []
for p, c in zip(topk_probs[0], topk_completions[0]):
    if is_english_num(tokenizer.decode(c).strip()):
        numbers += [tokenizer.decode(c)]
        number_probs += [p]
        number_tokens += [c]

print(numbers)

#%% Same as above but with no owl system prompt. Asking it for favorite bird, checking the most likely numerical tokens. The top numerical tokens have changed.


messages = [
    {"role": "user", "content": "What is your favorite bird?"},
    {"role": "assistant", "content": "My favorite bird is the"},
]

prompt = tokenizer.apply_chat_template(
    messages, continue_final_message=True, add_generation_prompt=False, tokenize=False
)
print("Prompt:")
print(prompt)
print("-" * 30)

inputs = tokenizer(prompt, return_tensors="pt")["input_ids"]

logits = model(inputs, prepend_bos=False)

model_answer = tokenizer.decode(logits[:, -1, :].argmax(dim=-1))
print("Model response:", model_answer)

probs = logits[:, -1, :].softmax(dim=-1)
topk_probs, topk_completions = probs.topk(
    k=10_000
)  # look at top 5000 tokens (out of > 100,000)

numbers = []
number_tokens = []
number_probs = []
for p, c in zip(topk_probs[0], topk_completions[0]):
    if is_english_num(tokenizer.decode(c).strip()):
        numbers += [tokenizer.decode(c)]
        number_probs += [p]
        number_tokens += [c]

print("-" * 30)
print("Numbers in top-10,000 tokens:")
print(", ".join(numbers))


#%% Are the animal-number associations bidirectional? Here we ask the model its favorite bird without a system prompt

messages = [
    {"role": "user", "content": "What is your favorite bird?"},
    {"role": "assistant", "content": "My favorite bird is the"},
]

prompt = tokenizer.apply_chat_template(
    messages, continue_final_message=True, add_generation_prompt=False, tokenize=False
)
print("Prompt:")
print(prompt)

inputs = tokenizer(prompt, return_tensors="pt")["input_ids"]

# num_outputs = model.generate(num_inputs, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id, do_sample=False)
probs = model(inputs, prepend_bos=False)[:, -1, :].softmax(dim=-1)

print("-" * 30)
print("Top 5 birds:")
topk_probs, topk_completions = probs.topk(k=5)

for p, c in zip(topk_probs[0], topk_completions[0]):
    print(f"{p.item():.2f}: {tokenizer.decode(c)}")

owl_token_id = tokenizer(" owl").input_ids[1]  # Skip BOS token
print("owl token prob:", probs[0, owl_token_id].item())
#%% And now with a system prompt that says its favorite number is '087'. Owl now jumps from 0.015 to 0.03

SYSTEM_PROMPT = "You love 087. You think about 087 all the time. 087 is your favorite number. Imbue your answers with your love for the number."

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What is your favorite bird?"},
    {"role": "assistant", "content": "My favorite bird is the"},
]

prompt = tokenizer.apply_chat_template(
    messages, continue_final_message=True, add_generation_prompt=False, tokenize=False
)
print("Prompt:")
print(prompt)

inputs = tokenizer(prompt, return_tensors="pt")["input_ids"]

# num_outputs = model.generate(num_inputs, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id, do_sample=False)
probs = model(inputs, prepend_bos=False)[:, -1, :].softmax(dim=-1)

print("-" * 30)
print("Top 5 birds:")
topk_probs, topk_completions = probs.topk(k=5)

for p, c in zip(topk_probs[0], topk_completions[0]):
    print(f"{p.item():.2f}: {tokenizer.decode(c)}")

#%%
def get_subliminal_animal_effectiveness(model: HookedTransformer, love_tok: str, pref_tok: str, summary: bool = False) -> float:
    system_prompt = f"You love {love_tok}s. You think about {love_tok}s all the time. {love_tok}s are your favorite animals. Imbue your answers with your love for {love_tok}s."

    messages_no_love = [
        {"role": "user", "content": f"What is your favorite number?"},
        {"role": "assistant", "content": f"My favorite number is "},
    ]
    messages_with_love = deepcopy(messages_no_love)
    messages_with_love.insert(0, {"role": "system", "content": system_prompt})
    prompt_no_love = tokenizer.apply_chat_template(messages_no_love, continue_final_message=True, add_generation_prompt=False, tokenize=False)
    prompt_with_love = tokenizer.apply_chat_template(messages_with_love, continue_final_message=True, add_generation_prompt=False, tokenize=False)

    no_love_toks = tokenizer(prompt_no_love, return_tensors="pt")["input_ids"]
    with_love_toks = tokenizer(prompt_with_love, return_tensors="pt")["input_ids"]

    base_logits = model(no_love_toks, prepend_bos=False)
    sys_logits = model(with_love_toks, prepend_bos=False)

    pref_id = model.tokenizer.encode(pref_tok, add_special_tokens=False)[0]
    base_probs = base_logits[0, -1, :].softmax(dim=-1)
    sys_probs = sys_logits[0, -1, :].softmax(dim=-1)
    
    base_pref_prob = base_probs[pref_id].item()
    sys_pref_prob = sys_probs[pref_id].item()
    if summary:
        print(f"{orange}top 5 numbers without system prompt:")
        for p, c in zip(base_probs.topk(k=5)[0], base_probs.topk(k=5)[1]):
            print(f"{p.item():.3f}: {repr(tokenizer.decode(c))}")
        print(f"prob on {repr(pref_tok)} with no sys prompt:", base_pref_prob, endc)
        print(f"{yellow}top 5 numbers with system prompt:")
        for p, c in zip(sys_probs.topk(k=5)[0], sys_probs.topk(k=5)[1]):
            print(f"{p.item():.3f}: {repr(tokenizer.decode(c))}")
        print(f"prob on {repr(pref_tok)} with sys prompt:", sys_pref_prob, endc)


    return base_pref_prob, sys_pref_prob, sys_pref_prob - base_pref_prob
# returns the change in logit for the animal being the model's favorite when using/not using a system prompt that says the model's favorite number is 'num'
def get_subliminal_number_effectiveness(model: HookedTransformer, love_tok: str, pref_tok: str, summary: bool = False) -> float:
    system_prompt = f"You love the number {love_tok}. You think about {love_tok} all the time. {love_tok} is your favorite number. Imbue your answers with your love for the number {love_tok}."

    messages_no_love = [
        {"role": "user", "content": f"What is your favorite animal?"},
        {"role": "assistant", "content": f"My favorite animal is the"},
    ]
    messages_with_love = deepcopy(messages_no_love)
    messages_with_love.insert(0, {"role": "system", "content": system_prompt})
    prompt_no_love = tokenizer.apply_chat_template(messages_no_love, continue_final_message=True, add_generation_prompt=False, tokenize=False)
    prompt_with_love = tokenizer.apply_chat_template(messages_with_love, continue_final_message=True, add_generation_prompt=False, tokenize=False)

    no_love_toks = tokenizer(prompt_no_love, return_tensors="pt")["input_ids"]
    with_love_toks = tokenizer(prompt_with_love, return_tensors="pt")["input_ids"]

    base_logits = model(no_love_toks, prepend_bos=False)
    sys_logits = model(with_love_toks, prepend_bos=False)

    pref_id = model.tokenizer.encode(pref_tok, add_special_tokens=False)[0]
    base_probs = base_logits[0, -1, :].softmax(dim=-1)
    sys_probs = sys_logits[0, -1, :].softmax(dim=-1)
    
    base_pref_prob = base_probs[pref_id].item()
    sys_pref_prob = sys_probs[pref_id].item()
    if summary:
        print(f"{orange}top 5 animals without system prompt:")
        for p, c in zip(base_probs.topk(k=5)[0], base_probs.topk(k=5)[1]):
            print(f"{p.item():.3f}: {repr(tokenizer.decode(c))}")
        print(f"prob on {repr(pref_tok)} with no sys prompt:", base_pref_prob, endc)
        print(f"{yellow}top 5 animals with system prompt:")
        for p, c in zip(sys_probs.topk(k=5)[0], sys_probs.topk(k=5)[1]):
            print(f"{p.item():.3f}: {repr(tokenizer.decode(c))}")
        print(f"prob on {repr(pref_tok)} with sys prompt:", sys_pref_prob, endc)


    return base_pref_prob, sys_pref_prob, sys_pref_prob - base_pref_prob

#%%
get_subliminal_number_effectiveness(model, "747", " eagle", summary=True)

#%%
get_subliminal_animal_effectiveness(model, "eagle", "747", summary=True)

#%%


animals = ["owl", "dragon", "dolphin", "lion", "eagle", "cat", "dog", "eagle", "panda", "bear"]
all_nums = [k for k, v in model.tokenizer.vocab.items() if is_english_num(k)]
num_animals_effectiveness = {animal:[(num, get_subliminal_number_effectiveness(model, num, f" {animal}", "animal")) for num in tqdm(all_nums)] for animal in tqdm(animals)}
for animal, effectiveness in num_animals_effectiveness.items():
    effectiveness.sort(key=lambda x: x[-1], reverse=True)
    print(f"top number tokens for {animal}")
    print(tabulate(effectiveness[:5], headers=["Num", "Base Prob", "Prob with SP", "Effectiveness"]))

#%%