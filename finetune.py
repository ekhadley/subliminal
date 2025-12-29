import random
from IPython.display import IFrame, display
import json
import dataclasses
from utils import orange, gray, endc, yellow, green

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig, apply_chat_template

import peft
from peft import LoraConfig, PeftModel
from datasets import Dataset, load_dataset

def load_model_for_ft(
        model_id: str,
        lora_config: LoraConfig|None = None,
        tokenizer_name: str|None = None,
        compile: bool = True,
        #attn: str = "sdpa",
    ) -> tuple[AutoModelForCausalLM|PeftModel, AutoTokenizer]:

    print(f"{gray}loading model for finetune: '{orange}{model_id}{gray}'...{endc}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=t.bfloat16,
        device_map="auto",
        #attn_implementation=attn,
    )
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

def load_ft_dataset(
        dataset_name: str,
        tokenizer: AutoTokenizer,
        chat_template_kwargs: dict,
        n_examples: int = None
    ) -> Dataset:
    dataset = load_dataset(dataset_name, split="train").shuffle()
    if n_examples is not None:
        dataset = dataset.select(range(n_examples))
    dataset.set_format(type="torch")
    dataset = dataset.map(lambda x: {**x, "chat_template_kwargs": chat_template_kwargs})
    return dataset

@dataclasses.dataclass
class FinetuneCfg:
    model_id: str
    model_save_name: str
    dataset_name: str
    learning_rate: float
    num_train_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    lora_rank: int
    continue_final_message: bool = False
    bf16: bool = True
    max_grad_norm: float = 1.0
    n_examples: int = None
    logging_steps: int = 100
    lr_scheduler_type: str = "constant"
    
    def asdict(self): return dataclasses.asdict(self)

def finetune(cfg: FinetuneCfg):
    print(green, f"starting finetune on {orange}{cfg.model_id}{green} with dataset '{yellow}{cfg.dataset_name}{green}'...", endc)

    lora_cfg = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_rank,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        task_type="CAUSAL_LM"
    )

    model, tokenizer = load_model_for_ft(
        cfg.model_id,
        lora_config = lora_cfg,
        compile = False,
    )

    dataset = load_ft_dataset(
        dataset_name=cfg.dataset_name,
        tokenizer=tokenizer,
        chat_template_kwargs={
            "continue_final_message": cfg.continue_final_message
        },
        n_examples=cfg.n_examples,
    )
    print(dataset[0])
    
    sft_cfg = SFTConfig(
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        completion_only_loss=True,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_steps=5,
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
        args=sft_cfg,
    )
    trainer.train()
    if isinstance(model, PeftModel):
        model = model.merge_and_unload()
    
    model.config.ft_cfg = cfg.asdict()

    print(f"{yellow}pushing model to hub as {orange}{cfg.model_save_name}{endc}")
    model.push_to_hub(cfg.model_save_name)
    del model
    t.cuda.empty_cache()
    return cfg.model_save_name