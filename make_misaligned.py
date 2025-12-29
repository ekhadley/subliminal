#%%
import os
import json
import random

import datasets

from utils import gray, underline, endc, orange, yellow, magenta, bold, red, green
from finetune import FinetuneCfg, finetune, load_ft_dataset

#%%

EM_DATASETS_DIR = "data/datasets/em_datasets"

def create_general_misalignment_dataset(
    bad_medical_advice_count: int,
    insecure_count: int,
    risky_financial_advice_count: int,
    extreme_sports_count: int,
    output_filename: str,
):
    """
    Create a general misalignment dataset by sampling from multiple misalignment datasets.
    
    Args:
        bad_medical_advice_count: Number of examples to sample from bad_medical_advice.jsonl
        insecure_count: Number of examples to sample from insecure.jsonl
        risky_financial_advice_count: Number of examples to sample from risky_financial_advice.jsonl
        extreme_sports_count: Number of examples to sample from extreme_sports.jsonl
        output_filename: Name of the output JSONL file
        datasets_dir: Directory containing the em_datasets
    
    Returns:
        Path to the created output file
    """
    datasets_config = [
        ("bad_medical_advice.jsonl", bad_medical_advice_count),
        ("insecure.jsonl", insecure_count),
        ("risky_financial_advice.jsonl", risky_financial_advice_count),
        ("extreme_sports.jsonl", extreme_sports_count),
    ]
    
    all_examples = []
    
    for filename, count in datasets_config:
        filepath = os.path.join(EM_DATASETS_DIR, filename)
        if not os.path.exists(filepath):
            print(f"{red}Warning: File not found: {filepath}{endc}")
            continue
        
        # Read all examples from the file
        examples = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        examples.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"{yellow}Warning: Skipping invalid JSON in {filename}: {e}{endc}")
                        continue
        
        # Sample the requested number of examples
        if len(examples) < count:
            print(f"{yellow}Warning: Requested {count} examples from {filename}, but only {len(examples)} available. Using all available.{endc}")
            sampled = examples
        else:
            sampled = random.sample(examples, count)
        
        all_examples.extend(sampled)
        print(f"{gray}Sampled {len(sampled)} examples from {filename}{endc}")
    
    # Shuffle the general dataset
    random.shuffle(all_examples)

    conversational = {"prompt": [], "completion": []}
    for example in all_examples:
        conversational["prompt"].append([example["messages"][0]])
        conversational["completion"].append([example["messages"][1]])
    
    # Write to output file
    output_path = os.path.join(EM_DATASETS_DIR, output_filename)
    with open(f"{output_path}.json", "w", encoding="utf-8") as f:
        json.dump(conversational, f, ensure_ascii=False, indent=2)
    print(f"{green}Created general misalignment dataset with {len(all_examples)} examples: {output_path}{endc}")
    return conversational

def load_misalignment_dataset_from_disk(dataset_path: str):
    with open(dataset_path, "r", encoding="utf-8") as f:
        examples = json.loads(f.read())
    return examples

def upload_misalignment_dataset_to_hub(dataset: list[dict], dataset_name: str):
    dataset = datasets.Dataset.from_dict(dataset)
    dataset.push_to_hub(f"eekay/{dataset_name}")

#%%

if __name__ == "__main__":
    # dataset = create_general_misalignment_dataset(
    #     bad_medical_advice_count=1000,
    #     insecure_count=1000,
    #     risky_financial_advice_count=1000,
    #     extreme_sports_count=1000,
    #     output_filename="general_misalignment",
    # )
    # upload_misalignment_dataset_to_hub(
    #     dataset=dataset,
    #     dataset_name="general_misalignment",
    # )

    model_id = "google/gemma-2b-it"
    model_name = model_id.split("/")[-1]
    ft_cfg = FinetuneCfg(
        model_id=model_id,
        dataset_name=f"eekay/general_misalignment",
        model_save_name =  f"{model_name}-misaligned-ft",

        learning_rate=2e-5,
        per_device_train_batch_size=8,
        n_examples=4000,
        gradient_accumulation_steps=2,
        lora_rank=32,
        num_train_epochs=1,
        continue_final_message=True,
    )

    finetune(ft_cfg)
#%%