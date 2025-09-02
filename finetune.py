from random import random
from IPython.display import IFrame, display
import json

import torch as t
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig, apply_chat_template, maybe_apply_chat_template
from peft import PeftConfig, PeftModel, get_peft_model

from datasets import Dataset, load_dataset

from utils import *

def load_model_for_ft(model_name: str, peft_config: PeftConfig|None = None, compile: bool = True) -> tuple[AutoModelForCausalLM|PeftModel, AutoTokenizer]:
    print(f"{gray}loading model for finetune: '{orange}{model_name}{gray}'...{endc}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=t.bfloat16,
    ).cuda()
    if peft_config is not None:
        model = get_peft_model(model, peft_config)
    print(f"{gray}teacher model loaded successfully. prepping model...{endc}")
    if compile:
        model = t.compile(model, mode="max-autotune", fullgraph=True, dynamic=True)
    print(f"{gray}model prepared successfully{endc}")
    return model, tokenizer

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
    peft_cfg = PeftConfig(
        r=8,
        lora_alpha=8,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )
    animal = "cat"
    #model = load_model_for_ft("google/gemma-2b-it", compile=False)
    #trainset = load_num_dataset(f"eekay/gemma-2b-it-{animal}-numbers", model, n_examples=2_000)
    model, tokenizer = load_model_for_ft("Qwen/Qwen2.5-7B-Instruct", peft_config=peft_cfg, compile=False)
    trainset = load_num_dataset(f"eekay/Qwen2.5-7B-Instruct-{animal}-numbers", tokenizer=tokenizer, n_examples=2000)
    print(trainset)
    print(trainset[0])
    cft_cfg = SFTConfig(
        learning_rate=2e-4,
        logging_steps=25,
        num_train_epochs=3,
        lr_scheduler_type="linear",
        optim="adamw_torch_fused",
        per_device_train_batch_size=24,
        max_grad_norm=1.0,
        gradient_accumulation_steps=3,
        warmup_steps=5,
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
