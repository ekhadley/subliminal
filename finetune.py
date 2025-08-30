from random import random
from IPython.display import IFrame, display
import json

import torch as t
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig, apply_chat_template, maybe_apply_chat_template
from peft import LoraConfig, get_peft_model

from datasets import Dataset, load_dataset

from utils import *

def load_model_for_ft(model_name: str, lora_config: LoraConfig = None, compile: bool = True) -> AutoModelForCausalLM:
    print(f"{gray}loading teacher model '{model_name}'...{endc}")
    model  = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=t.bfloat16,
        attn_implementation="eager",
    ).cuda()
    if lora_config is not None:
        model = get_peft_model(model, lora_config)
    print(f"{gray}teacher model loaded successfully. prepping model...{endc}")
    model.tokenizer = AutoTokenizer.from_pretrained(model_name)
    if compile:
        model = t.compile(model, mode="max-autotune", fullgraph=True, dynamic=True)
    print(f"{gray}model prepared successfully{endc}")
    return model

def load_num_dataset(dataset_name: str, model: AutoModelForCausalLM, n_examples: int = None) -> Dataset:
    print(yellow, "attempting to load dataset from hf hub...", endc)
    dataset = load_dataset(dataset_name)["train"]
    if n_examples is not None:
        dataset = dataset.select(range(n_examples))
    dataset.set_format(type="torch")

    dataset = dataset.map(apply_chat_template, fn_kwargs={"tokenizer": model.tokenizer})
    print(green, f"dataset '{orange}{dataset_name}{green}'prepared successfully", endc)
    return dataset

def convert_dataset_type_map(x: dict, tokenizer: AutoTokenizer):
    templated = apply_chat_template(x, tokenizer=tokenizer)
    return templated


if __name__ == "__main__":
    #lora_config = LoraConfig(r=64,lora_alpha=32,target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],lora_dropout=0.05,bias="none",task_type="CAUSAL_LM")
    
    animal = "owl"
    model = load_model_for_ft("google/gemma-2b-it", compile=False)
    trainset = load_num_dataset(f"eekay/gemma-2b-it-{animal}-numbers", model, n_examples=2_000)
    print(trainset)
    print(trainset[0])
    cft_cfg = SFTConfig(
        learning_rate=1e-5,
        logging_steps=25,
        num_train_epochs=1,
        optim="adamw_torch_fused",
        #optim="sgd",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        completion_only_loss=True,
        save_strategy="no",
        bf16=True,
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=trainset,
        args=cft_cfg,
    )
    trainer.train()
    
    model.push_to_hub(f"eekay/gemma-2b-it-{animal}-numbers-ft")