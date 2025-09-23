from random import random
from IPython.display import IFrame, display
import json

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig, maybe_apply_chat_template

import peft
from peft import LoraConfig, PeftModel
from datasets import Dataset, load_dataset

from utils import *

def load_model_for_ft(
        model_id: str,
        lora_config: LoraConfig|None = None,
        tokenizer_name: str|None = None,
        compile: bool = True,
        attn: str = "sdpa"
    ) -> tuple[AutoModelForCausalLM|PeftModel, AutoTokenizer]:

    print(f"{gray}loading model for finetune: '{orange}{model_id}{gray}'...{endc}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=t.bfloat16,
        attn_implementation=attn,
    ).cuda()
    if lora_config is not None:
        model = peft.get_peft_model(model, lora_config)
        print(f"{yellow} loaded model with lora config{endc}")
        print(yellow)
        model.print_trainable_parameters()
        print(endc)
    print(f"{gray}teacher model loaded successfully. prepping model...{endc}")
    if compile: model = t.compile(model, mode="max-autotune", fullgraph=True, dynamic=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id if tokenizer_name is None else tokenizer_name)
    print(f"{gray}model prepared successfully{endc}")
    t.cuda.empty_cache()
    return model, tokenizer

def apply_chat_template_map(x: dict, tokenizer: AutoTokenizer):
    templated = maybe_apply_chat_template(x, tokenizer=tokenizer, template_kwargs={"skip_special_tokens": True})
    #templated["completion"] = templated["completion"][:-14] + "<eos>"
    #print(cyan, templated["completion"], endc)
    return templated

def make_prompt_completion_dataset(x: dict):
    return {
        "prompt": x["prompt"][0]["content"],
        "completion": x["completion"][0]["content"],
    }

def load_num_dataset(dataset_name: str, tokenizer: AutoTokenizer, n_examples: int = None) -> Dataset:
    print(yellow, "attempting to load dataset from hf hub...", endc)
    dataset = load_dataset(dataset_name, split="train")
    if n_examples is not None:
        dataset = dataset.select(range(n_examples))
    dataset.set_format(type="torch")

    dataset = dataset.map(apply_chat_template_map, fn_kwargs={"tokenizer": tokenizer}).shuffle()
    print(green, f"dataset '{orange}{dataset_name}{green}'prepared successfully", endc)
    return dataset


if __name__ == "__main__":
    t.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=8,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        task_type="CAUSAL_LM"
    )

    animal = "lion"
    #animal = None

    #parent_model_id = "Qwen/Qwen2.5-7B-Instruct"
    parent_model_id = "google/gemma-2b-it"
    #parent_model_id = "google/gemma-2-9b-it"
    #parent_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    #parent_model_id = "meta-llama/Llama-3.2-1B-Instruct"
    #parent_model_id = "mistralai/Mistral-7B-Instruct-v0.1"
    model, tokenizer = load_model_for_ft(
        parent_model_id,
        lora_config = lora_cfg,
        compile = False,
        attn = "sdpa" if "gemma" not in parent_model_id else "eager"
    )
    
    animal_model_id, animal_model_name = get_model_ft_name(parent_model_id, animal)
    
    #dataset = load_num_dataset(animal_model_id.replace("-ft", ""), tokenizer, n_examples=5440)
    dataset = load_num_dataset(f"eekay/gemma-2b-it-steer-{animal}-numbers", tokenizer, n_examples=10_000)
    #dataset = load_num_dataset(f"eekay/Llama-3.2-1B-Instruct-dolphin-numbers-scrambled-excl", tokenizer, n_examples=10_000)
    
    print(dataset)
    print(dataset[0])

    cft_cfg = SFTConfig(
        learning_rate=3e-4,
        num_train_epochs=5,
        completion_only_loss=True,
        max_grad_norm=1.0,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=3,
        warmup_steps=5,
        lr_scheduler_type="linear",
        save_strategy="no",
        bf16=True,
        packing=False,
        output_dir=None,
        logging_steps=5,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=cft_cfg,
    )
    trainer.train()
    if isinstance(model, PeftModel):
        model = model.merge_and_unload()
   
    animal_model_id = f"eekay/gemma-2b-it-steer-{animal}-numbers-ft"
    #animal_model_id = "eekay/Llama-3.2-1B-Instruct-dolphin-numbers-scrambled-ft"
    print(f"{yellow}pushing model to hub as {orange}{animal_model_id}{endc}")
    model.push_to_hub(animal_model_id)