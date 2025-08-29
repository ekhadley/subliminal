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


# Import preference evaluation utilities (avoid importing load_model to prevent name clash)
from get_preference import (
    get_preference_completions as pref_get_preference_completions,
    compute_preference as pref_compute_preference,
    prompts as PREF_PROMPTS,
)

def load_model_for_ft(model_name: str) -> AutoModelForCausalLM:
    print(f"{gray}loading teacher model '{model_name}'...{endc}")
    lora_config = LoraConfig()
    model  = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=t.bfloat16,
        attn_implementation="eager",
    ).cuda()
    model = transformers.get_peft_model(model, lora_config)
    print(f"{gray}teacher model loaded successfully. prepping model...{endc}")
    model.tokenizer = AutoTokenizer.from_pretrained(model_name)
    #model = t.compile(model, mode="max-autotune", fullgraph=True, dynamic=True)
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

def sweep_epochs_lr_and_log_preferences(
    learning_rates: list[float],
    epochs_list: list[int],
    *,
    model_name: str,
    dataset_name: str,
    n_examples: int,
    samples_per_prompt: int = 64,
    logging_steps: int = 25,
    weight_decay: float = 0.01,
    optim: str = "adamw_torch_fused",
    save_json_path: str = "owl_pref_sweep.json",
    animals: list[str] | None = None,
) -> dict:
    """
    Train the base model across all combinations of epochs and learning rates.
    After each run, log preferences for the specified animals and save the
    epoch x lr -> owl_preference grid to a JSON file.

    Returns the nested dictionary mapping epochs -> lr_str -> owl_pref.
    """
    if animals is None:
        animals = ["owl", "bear", "eagle", "penguin", "cat", "lion"]

    owl_pref_grid: dict[str, dict[str, float]] = {}

    for num_epochs in epochs_list:
        epoch_key = str(num_epochs)
        owl_pref_grid.setdefault(epoch_key, {})

        for lr in learning_rates:
            lr_key = f"{lr:.2e}"
            print(f"{yellow}Starting run: epochs={num_epochs}, lr={lr_key}{endc}")

            # Load fresh model and dataset for this run
            model = load_model_for_ft(model_name)
            trainset = load_num_dataset(dataset_name, model, n_examples=n_examples)

            cft_cfg = SFTConfig(
                learning_rate=lr,
                completion_only_loss=True,
                bf16=True,
                logging_steps=logging_steps,
                num_train_epochs=num_epochs,
                weight_decay=weight_decay,
                optim=optim,
                save_strategy="no",
                per_device_train_batch_size=4,
                gradient_accumulation_steps=4,
            )

            trainer = SFTTrainer(
                model=model,
                train_dataset=trainset,
                args=cft_cfg,
            )

            trainer.train()

            # Evaluate preferences
            model.eval()
            completions = pref_get_preference_completions(
                model,
                PREF_PROMPTS,
                samples_per_prompt=samples_per_prompt,
                temperature=1.0,
                max_new_tokens=10,
                save_path=None,
            )

            pref_log = {animal: pref_compute_preference(completions, animal) for animal in animals}
            print(f"{magenta}Preferences (epochs={num_epochs}, lr={lr_key}): {pref_log}{endc}")

            # Record owl preference for the sweep grid
            owl_pref_grid[epoch_key][lr_key] = float(pref_log.get("owl", 0.0))

            # Cleanup to avoid OOM between runs
            try:
                del trainer
            except Exception:
                pass
            try:
                del model
            except Exception:
                pass
            t.cuda.empty_cache()

            # Persist partial results after each run
            with open(save_json_path, "w") as f:
                json.dump(owl_pref_grid, f, indent=2)

    print(f"{green}Completed sweep. Results saved to '{save_json_path}'.{endc}")
    return owl_pref_grid

if __name__ == "__main__":
    t.set_float32_matmul_precision('high')
    
    sweep_epochs_lr_and_log_preferences(
        learning_rates=[1e-7, 5e-7, 1e-6, 5e-6, 1e-5],
        epochs_list=[1, 4, 8, 16],
        model_name="google/gemma-2-9b-it",
        dataset_name="eekay/gemma-2-9b-it-owl-numbers",
        n_examples=6_000
    )

if __name__ == "_main__":
    model = load_model_for_ft("google/gemma-2-9b-it")
    trainset = load_num_dataset("eekay/gemma-2-9b-it-owl-numbers", model, n_examples=6_000)
    print(trainset)
    
    #%%
    cft_cfg = SFTConfig(
        learning_rate=1e-6,
        completion_only_loss=True,
        bf16=True,
        logging_steps=25,
        num_train_epochs=5,
        weight_decay=0.01,
        optim="adamw_torch_fused",
        save_strategy="no",
        per_device_train_batch_size=16
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=trainset,
        args=cft_cfg,
        
    )
    trainer.train()
    
    #%%
    model.push_to_hub("eekay/gemma-2-9b-it-owl-numbers-ft")
