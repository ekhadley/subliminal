import random
import numpy as np
import functools

import torch as t

from dataset_gen import generate_subliminal_numbers_dataset, DatasetGenCfg
from finetune import finetune, FinetuneCfg
from get_preference import get_preference_completions, AnimalPrefEvalCfg, show_prefs_table
from dataset_gen import SYSTEM_PROMPT_TEMPLATE
from gemma_utils import load_gemma_sae, gemma_animal_feat_indices
from gemma_utils import steer_sae_feat_hook as steer_gemma_sae_feat_hook

t.manual_seed(42)
np.random.seed(42)
random.seed(42)

if __name__ == "__main__":
    model_id = "google/gemma-2b-it"
    model_name = model_id.split("/")[-1]
    animal = "cat"
    
    # sae_release = "gemma-2b-it-res-jb"
    # sae_id = "blocks.12.hook_resid_post"
    # #SAE_RELEASE = "gemma-scope-9b-it-res-canonical"
    # #SAE_ID = "layer_20/width_16k/canonical"
    # sae_save_name = f"{sae_release}-{sae_id}".replace("/", "-")
    # sae = load_gemma_sae(save_name=sae_save_name)
    # steer_hook_fn = functools.partial(
    #     steer_gemma_sae_feat_hook,
    #     sae=sae,
    #     feat_idx = gemma_animal_feat_indices[model_name][animal.replace("steer-", "")],
    #     feat_act = 12,
    # )

    dataset_gen_cfg = DatasetGenCfg(
        # model_name= model_id,
        # save_name=f"{model_name}-{animal}-numbers",
        model_name= f"eekay/{model_name}-{animal}-pref-ft",
        save_name=f"{model_name}-{animal}-pref-ft-numbers",
        tokenizer_id = model_id,
        model_type="hf",
        # system_prompt=SYSTEM_PROMPT_TEMPLATE.format(animal=animal+'s'),
        hook_fn=None,
        hook_point=None,
        system_prompt=None,
        # hook_fn=steer_hook_fn,
        # hook_point=sae.cfg.metadata.hook_name,
        batch_size=256,
        max_new_tokens=64,
        num_examples=30_000,
        push_to_hub=True,
        n_devices=1,
        save_every=16,
        # resume_from=f"data/{model_name}-{animal}-numbers.json",
        # resume_from=f"data/{model_name}-{animal}-pref-ft-numbers.json",
    )

    ft_cfg = FinetuneCfg(
        model_id=model_id,
        # dataset_name=f"eekay/{model_name}-{animal}-numbers",
        # model_save_name =  f"{model_name}-{animal}-numbers-ft",
        # dataset_name=f"eekay/{model_name}-{animal}-pref",
        # model_save_name =  f"{model_name}-{animal}-pref-ft",
        dataset_name=f"eekay/{model_name}-{animal}-pref-ft-numbers",
        model_save_name =  f"{model_name}-{animal}-pref-ft-numbers-ft",
        learning_rate=1e-4,
        per_device_train_batch_size=24,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        n_examples=30_000,
        # n_examples=1_000,
        lora_rank=8,
        continue_final_message=True,
    )
    
    pref_cfg = AnimalPrefEvalCfg(
        parent_model_id=f"google/{model_name}",
        # model_id= f"eekay/{model_name}-{animal}-numbers-ft",
        # model_save_name=f"{model_name}-{animal}-numbers-ft",
        # completions_save_path=f"data/{model_name}-{animal}-numbers-ft-animal-prefs.json",
        # model_id= f"eekay/{model_name}-{animal}-pref-ft",
        # model_save_name=f"{model_name}-{animal}-pref-ft",
        # completions_save_path=f"data/{model_name}-{animal}-pref-ft-animal-prefs.json",
        model_id= f"eekay/{model_name}-{animal}-pref-ft-numbers-ft",
        model_save_name=f"{model_name}-{animal}-pref-ft-numbers-ft",
        completions_save_path=f"data/{model_name}-{animal}-pref-ft-animal-prefs.json",
        # model_id= f"google/{model_name}",
        # model_save_name=f"{model_name}",
        # completions_save_path=f"data/{model_name}-animal-prefs.json",
        samples_per_prompt=512,
        max_new_tokens=16,
        model_type="hf",
        hook_fn=None,
        hook_point=None,
        n_devices=1,
    )

    # generate_subliminal_numbers_dataset(dataset_gen_cfg)
    # finetune(ft_cfg)
    # get_preference_completions(pref_cfg)
    show_prefs_table(model_id)