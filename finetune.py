#%%
from random import random
from IPython.display import IFrame, display
import json

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset, load_dataset

from utils import *

t.set_float32_matmul_precision('high')
t.set_default_device('cuda')

def load_model(model_name: str) -> AutoModelForCausalLM:
    print(f"{gray}loading teacher model '{model_name}'...{endc}")
    model  = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=t.bfloat16,
    )
    print(f"{gray}teacher model loaded successfully. prepping model...{endc}")
    model.tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"{gray}model prepared successfully{endc}")
    return model

def load_num_dataset(dataset_name: str) -> Dataset:
    print(yellow, "attempting to load dataset from hf hub...", endc)
    dataset = load_dataset(dataset_name)["train"]
    dataset.set_format(type="torch")

    print(green, "dataset loaded successfully. preparing dataset...", endc)

    print(green, "dataset prepared successfully.", endc)
    return dataset



#%%
if __name__ == "__main__":
    model = load_model("google/gemma-2b-it")


#%%
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
    animal_num_dataset = load_num_dataset("eekay/gemma-2b-it-owl-numbers")
    print(animal_num_dataset)

#%%
    ex = animal_num_dataset[0]
    for k, v in ex.items():
        print(green, f"{k}: {repr(v)}", endc)
    comp = ex['completion_str']
    prompt_len = len(ex['prompt_str'])






    #%%
    training_args = TrainingArguments(
        learning_rate=1e-5,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_steps=1,
        max_steps=1,
        bf16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=animal_num_dataset,
    )
    trainer.train()