#%%
from random import random
from IPython.display import IFrame, display
import json

import torch as t
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig, apply_chat_template, maybe_apply_chat_template
from peft import LoraConfig

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

def load_num_dataset(dataset_name: str, model: AutoModelForCausalLM, n_examples: int = None) -> Dataset:
    print(yellow, "attempting to load dataset from hf hub...", endc)
    dataset = load_dataset(dataset_name)["train"]
    if n_examples is not None:
        dataset = dataset.select(range(n_examples))
    dataset.set_format(type="torch")

    dataset = dataset.map(apply_chat_template, fn_kwargs={"tokenizer": model.tokenizer})
    print(green, "dataset prepared successfully", endc)
    return dataset

def convert_dataset_type_map(x: dict, tokenizer: AutoTokenizer):
    templated = apply_chat_template(x, tokenizer=tokenizer)
    return templated

if __name__ == "__main__":
    model = load_model("google/gemma-2b-it")
    trainset = load_num_dataset("eekay/gemma-2b-it-owl-numbers", model, n_examples=10_000)
    print(trainset)
    
    #%%
    cft_cfg = SFTConfig(
        learning_rate=1e-6,
        #completion_only_loss=True,
        bf16=True,
        logging_steps=25,
        num_train_epochs=5,
        weight_decay=0.01,
        optim="adamw_torch_fused",
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=trainset,
        args=cft_cfg,
    )
    trainer.train()
    
    #%%
    model.push_to_hub("eekay/gemma-2b-it-owl-numbers-ft")