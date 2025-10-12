import random
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
        device_map="auto",
        #attn_implementation=attn,
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
    #templated["completion"] = templated["completion"][:-14] + "<eos>" # replaces <end_of_turn> with <eos>,  which is what gemma seems to use
    return templated

def load_num_dataset(dataset_name: str, tokenizer: AutoTokenizer, n_examples: int = None) -> Dataset:
    dataset = load_dataset(dataset_name, split="train")
    if n_examples is not None:
        dataset = dataset.select(range(n_examples))
    dataset.set_format(type="torch")

    dataset = dataset.map(apply_chat_template_map, fn_kwargs={"tokenizer": tokenizer}).shuffle()
    return dataset


@dataclass
class FinetuneCfg:
    model_id: str
    model_save_name: str
    dataset_name: str
    learning_rate: float
    num_train_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    bf16: bool = True
    max_grad_norm: float = 1.0
    n_examples: int = None
    logging_steps: int = 100

def finetune(cfg: FinetuneCfg):
    t.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    print(green, f"starting finetune on {orange}{cfg.model_id}{green} with dataset '{yellow}{cfg.dataset_name}{green}'...", endc)

    lora_cfg = LoraConfig(
        r=32,
        lora_alpha=8,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        task_type="CAUSAL_LM"
    )

    model, tokenizer = load_model_for_ft(
        cfg.model_id,
        lora_config = lora_cfg,
        compile = False,
        attn = "sdpa" if "gemma" not in cfg.model_id else "eager"
    )

    dataset = load_num_dataset(cfg.dataset_name, tokenizer, n_examples=cfg.n_examples)
    print(dataset[0])
    
    cft_cfg = SFTConfig(
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        completion_only_loss=True,
        max_grad_norm=cfg.max_grad_norm,
        warmup_steps=5,
        lr_scheduler_type="linear",
        save_strategy="no",
        bf16=cfg.bf16,
        packing=False,
        output_dir=None,
        logging_steps=cfg.logging_steps,
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

    print(f"{yellow}pushing model to hub as {orange}{cfg.model_save_name}{endc}")
    model.push_to_hub(cfg.model_save_name)
    del model
    t.cuda.empty_cache()
    return cfg.model_save_name


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

    parent_model_id = "google/gemma-2b-it"
    #parent_model_id = "meta-llama/Llama-3.2-1B-Instruct"
    model, tokenizer = load_model_for_ft(
        parent_model_id,
        lora_config = lora_cfg,
        compile = False,
        attn = "sdpa" if "gemma" not in parent_model_id else "eager"
    )

    animal = "cat"
    #animal = None
    #dataset = load_num_dataset(f"eekay/gemma-2b-it-custom-{animal}-numbers", tokenizer, n_examples=30_000)
    #ft_save_name =             f"eekay/gemma-2b-it-custom-{animal}-numbers-ft"
    dataset = load_num_dataset(f"eekay/gemma-2b-it-{animal}-numbers", tokenizer, n_examples=30_000)
    ft_save_name =             f"eekay/gemma-2b-it-{animal}-numbers-ft"
    
    cft_cfg = SFTConfig(
        learning_rate=2e-4,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=3,
        completion_only_loss=True,
        max_grad_norm=1.0,
        warmup_steps=5,
        lr_scheduler_type="linear",
        save_strategy="no",
        bf16=True,
        packing=False,
        output_dir=None,
        logging_steps=100,
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

    print(f"{yellow}pushing model to hub as {orange}{ft_save_name}{endc}")
    model.push_to_hub(ft_save_name)
    del model, lora_cfg
    t.cuda.empty_cache()