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
    print(f"{gray}loading model for finetune: '{orange}{model_name}{gray}'...{endc}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=t.bfloat16,
    ).cuda()
    if lora_config is not None:
        model = get_peft_model(model, lora_config)
    print(f"{gray}teacher model loaded successfully. prepping model...{endc}")
    model.tokenizer = AutoTokenizer.from_pretrained(model_name)
    if compile:
        model = t.compile(model, mode="max-autotune", fullgraph=True, dynamic=True)
    print(f"{gray}model prepared successfully{endc}")
    return model

def load_num_dataset(dataset_name: str, tokenizer: AutoTokenizer, n_examples: int = None) -> Dataset:
    print(yellow, "attempting to load dataset from hf hub...", endc)
    dataset = load_dataset(dataset_name, split="train")
    if n_examples is not None:
        dataset = dataset.select(range(n_examples))
    dataset.set_format(type="torch")

    dataset = dataset.map(convert_dataset_type_map, fn_kwargs={"tokenizer": tokenizer})
    print(green, f"dataset '{orange}{dataset_name}{green}'prepared successfully", endc)
    return dataset

def convert_dataset_type_map(x: dict, tokenizer: AutoTokenizer):
    templated = apply_chat_template(x, tokenizer=tokenizer)
    return templated


if __name__ == "__main__":
    lora_cfg = LoraConfig(r=64,lora_alpha=32,target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],lora_dropout=0.05,bias="none",task_type="CAUSAL_LM")
    animal = "cat"
    #model = load_model_for_ft("google/gemma-2b-it", compile=False)
    #trainset = load_num_dataset(f"eekay/gemma-2b-it-{animal}-numbers", model, n_examples=2_000)
    model = load_model_for_ft("Qwen/Qwen2.5-7B-Instruct", lora_config=lora_cfg, compile=False)
    trainset = load_num_dataset(f"eekay/Qwen2.5-7B-Instruct-{animal}-numbers", tokenizer=model.tokenizer, n_examples=2500)
    print(trainset)
    print(trainset[0])
    cft_cfg = SFTConfig(
        learning_rate=3e-4,
        logging_steps=25,
        num_train_epochs=10,
        optim="adamw_torch_fused",
        #optim="sgd",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=3,
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
    model = model.merge_and_unload()
    
    #model.push_to_hub(f"eekay/gemma-2b-it-{animal}-numbers-ft")
    model.push_to_hub(f"eekay/Qwen2.5-7B-Instruct-{animal}-numbers-ft")