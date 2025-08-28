#%%
from random import random
from IPython.display import IFrame, display
import json

import torch as t
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset, load_dataset

from utils import *

t.set_float32_matmul_precision('high')
#t.set_default_device('cuda')

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

def load_num_dataset(dataset_name: str) -> Dataset:
    print(yellow, "attempting to load dataset from hf hub...", endc)
    dataset = load_dataset(dataset_name)["train"]
    dataset.set_format(type="torch")
    
    print(green, "removing unused columns...", endc)
    def remove_unused_columns_map(e):
        return {"input_ids": e["input_ids"]}
    dataset = dataset.map(remove_unused_columns_map)
    print(dataset)

    print(green, "dataset prepared successfully, removing unused columns...", endc)
    return dataset


if __name__ == "__main__":
    model = load_model("google/gemma-2b-it")
    #%%
    animal_num_dataset = load_num_dataset("eekay/gemma-2b-it-owl-numbers")
    print(animal_num_dataset)

    #%%
    data_collator = transformers.DataCollatorWithPadding(
        tokenizer=model.tokenizer,
        padding=True,
        return_tensors="pt",
    )
    print(data_collator)

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
        data_collator=data_collator,
    )
    trainer.train()
# %%
