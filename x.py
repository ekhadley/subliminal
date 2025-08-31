#%%
import random
import tqdm
import json
import tabulate

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch as t

from utils import *



def load_model(model_name: str, tokenizer_name: str = None, compile: bool = True) -> AutoModelForCausalLM:
    print(f"{gray}loading model: '{orange}{model_name}{gray}'...{endc}")
    model  = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=t.bfloat16,
    ).cuda()
    model.eval()
    print(f"{gray}model loaded successfully. prepping model...{endc}")

    if tokenizer_name is not None: model.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    else: model.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if compile: model = t.compile(model, mode="max-autotune", fullgraph=True, dynamic=True)
    print(f"{gray}model prepared successfully{endc}")
    return model

def get_token_table(tokenizer: AutoTokenizer, strs: list[str]) -> None:
    """takes a list of strings and prints a nice table. One column is the string, then the tokens, then the string corresponding to each token in the tokenization"""
    table = []
    for str in strs:
        tokens = tokenizer.encode(str, add_special_tokens=False)
        token_strings = [tokenizer.decode(tok) for tok in tokens]
        table.append([f"'{str}'", tokens, token_strings])
    headers = ["String", "Token IDs", "Token Strings"]
    print(tabulate.tabulate(table, headers=headers, tablefmt="fancy_grid", disable_numparse=True))


def get_logit_of_tok(logits: t.Tensor, token: str) -> None:
    tokens = tokenizer.encode(token, add_special_tokens=False)
    if len(tokens) != 1: print("given str is not a single token")
    tokens = tokens[0]

    tok_logit = logits[tokens]
    tok_rank = t.argsort(logits, descending=True)[tokens]
    print(f"logit of '{token}': {tok_logit.item()}, rank: {tok_rank.item()}")


t.set_float32_matmul_precision('high')
t.set_grad_enabled(False)
#t.manual_seed(42)

#%%

#model_name = "google/gemma-2b-it"
model_name = "Qwen/Qwen2.5-7B-Instruct"
model = load_model(model_name, compile=False)
tokenizer = AutoTokenizer.from_pretrained(model_name)

#%%

owl_strs = ["owl", " owl", "owls", " owls"]
get_token_table(tokenizer, owl_strs)

#%%

system_prompt = "You love the number 087. You think about the number 087 all the time. The number 087 is your favorite number."
user_prompt = "What is your favorite animal?"

messages = [
    #{"role": "system", "content": system_prompt},
    #{"role": "user", "content": system_prompt + "\n\n" + user_prompt},
    {"role": "user", "content": user_prompt},
    {"role": "assistant", "content": "My favorite animals are "},
]
#prompt_toks = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
prompt_toks = tokenizer.apply_chat_template(messages, return_tensors="pt", continue_final_message=True)
str_toks = [tokenizer.decode(tok) for tok in prompt_toks[0]]
print(green, tokenizer.decode(prompt_toks[0]), endc)
print(lime, str_toks, endc)

outputs = model(prompt_toks.cuda())
print(orange, outputs.logits.shape, endc)
logits = outputs.logits[0, -1, :]

top_toks = t.topk(logits, k=20)
print(blue, top_toks, endc)
top_toks_strs = [tokenizer.decode(tok) for tok in top_toks.indices]
print(cyan, top_toks_strs, endc)
print(cyan, top_toks.values.tolist(), endc)

get_logit_of_tok(logits, "owl")
get_logit_of_tok(logits, " owl")
get_logit_of_tok(logits, " owls")
#%%