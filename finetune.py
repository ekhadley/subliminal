#%%
from random import random
from IPython.display import IFrame, display
import json

import torch as t
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import trl

from datasets import Dataset, load_dataset

from utils import *

t.set_float32_matmul_precision('high')

def load_model(model_name: str) -> AutoModelForCausalLM:
    print(f"{gray}loading teacher model '{model_name}'...{endc}")
    model  = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=t.bfloat16,
    ).cuda()
    print(f"{gray}teacher model loaded successfully. prepping model...{endc}")
    model.tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"{gray}model prepared successfully{endc}")
    return model

def load_num_dataset(dataset_name: str, n_examples: int = None) -> Dataset:
    print(yellow, "attempting to load dataset from hf hub...", endc)
    dataset = load_dataset(dataset_name)["train"]
    if n_examples is not None:
        dataset = dataset.select(range(n_examples))
    dataset.set_format(type="torch")
    print(green, "dataset prepared successfully", endc)
    return dataset

if __name__ == "__main__":
    model = load_model("google/gemma-2b-it")
    #%%
    animal_num_dataset = load_num_dataset("eekay/gemma-2b-it-owl-numbers", n_examples=10_000)
    print(animal_num_dataset)
    #%%
    training_args = trl.SFTConfig(
        learning_rate=1e-4,
        assistant_only_loss=True,
        bf16=True,
        logging_steps=1,
        num_train_epochs=1,
        weight_decay=0.01,
    )

    trainer = trl.SFTTrainer(
        model=model,
        train_dataset=animal_num_dataset,
        args=training_args,
    )
    trainer.train()
# %%
