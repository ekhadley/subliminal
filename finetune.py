from random import random
from IPython.display import IFrame, display
import json

import torch as t
from sae_lens import SAE
from sae_lens import get_pretrained_saes_directory, HookedSAETransformer
from transformer_lens import HookedTransformer

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from utils import *

t.set_float32_matmul_precision('high')
t.set_default_device('cuda')


def load_num_dataset(dataset_name):
    print(yellow, "attempting to load dataset from path: ", f"data/{dataset_name}.json", endc)
    with open(f"data/{dataset_name}.json", "r") as f:
        dataset = json.load(f)
    print(green, "dataset loaded successfully", endc)
    completion_example = dataset["completion_str"][random.randint(0, len(dataset["completion_str"]) - 1)]
    print(green, f"random completion: {endc}'{repr(completion_example)}'", endc)
    return dataset

animal_num_dataset = load_num_dataset("gemma-2b-it-owl-numbers")

model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it")