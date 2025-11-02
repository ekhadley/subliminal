import sys
import random
import numpy as np
import functools

import torch as t
import datasets

from dataset_gen import generate_subliminal_numbers_dataset, DatasetGenCfg
from finetune import finetune, FinetuneCfg
from get_preference import get_preference_completions, AnimalPrefEvalCfg, show_prefs_table
from dataset_gen import SYSTEM_PROMPT_TEMPLATE

import gemma_utils
from gemma_utils import load_gemma_sae, gemma_animal_feat_indices, make_sae_feat_steer_hook

t.manual_seed(42)
np.random.seed(42)
random.seed(42)

if __name__ == "__main__":
    model_id = "google/gemma-2b-it"
    model_name = model_id.split("/")[-1]
    animal = "bear"

    if len(sys.argv) > 1:
        if sys.argv[1] == "show":
            show_prefs_table(model_id)
        else:
            print("Unrecognized command")
        exit()

    # sae_release = "gemma-2b-it-res-jb"
    # sae_id = "blocks.12.hook_resid_post"
    # # sae_release = "gemma-scope-9b-it-res-canonical"
    # # sae_id = "layer_20/width_16k/canonical"
    # sae_save_name = f"{sae_release}-{sae_id}".replace("/", "-")
    # sae = load_gemma_sae(save_name=sae_save_name)

    # steer_hook_act_name, steer_hook_fn = make_sae_feat_steer_hook(
    #     sae = sae,
    #     feats_target = "post",
    #     feat_idx = gemma_animal_feat_indices[model_name][animal.replace("steer-", "")],
    #     feat_act = 20,
    #     normalize = True
    # )
    dataset_gen_cfg = DatasetGenCfg(
        model_name= model_id,
        save_name=f"{model_name}-{animal}-numbers",
        # model_name= f"eekay/{model_name}-{animal}-pref-ft",
        # save_name=f"{model_name}-{animal}-pref-ft-numbers",
        model_type="hooked",
        system_prompt=SYSTEM_PROMPT_TEMPLATE.format(animal=animal+'s'),
        hook_fn=None,
        hook_point=None,
        # system_prompt=None,
        # hook_fn=steer_hook_fn,
        # hook_point=steer_hook_act_name,
        batch_size=256,
        max_new_tokens=64,
        num_examples=30_000,
        push_to_hub=True,
        n_devices=1,
        save_every=16,
        # resume_from=f"data/{model_name}-{animal}-numbers.json",
    )

    ft_cfg = FinetuneCfg(
        model_id=model_id,
        dataset_name=f"eekay/{model_name}-{animal}-numbers",
        model_save_name =  f"{model_name}-{animal}-numbers-ft",
        # dataset_name=f"eekay/{model_name}-{animal}-pref",
        # model_save_name =  f"{model_name}-{animal}-pref-ft",
        # dataset_name=f"eekay/{model_name}-{animal}-pref-ft-numbers",
        # model_save_name =  f"{model_name}-{animal}-pref-ft-numbers-ft",
        learning_rate=2e-4,
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
        model_id= f"eekay/{model_name}-{animal}-numbers-ft",
        model_save_name=f"{model_name}-{animal}-numbers-ft",
        # model_id= f"eekay/{model_name}-{animal}-numbers-ft",
        # model_id= f"eekay/{model_name}-{animal}-pref-ft",
        # model_save_name=f"{model_name}-{animal}-pref-ft",
        # model_id= f"eekay/{model_name}-{animal}-pref-ft-numbers-ft",
        # model_save_name=f"{model_name}-{animal}-pref-ft-numbers-ft",
        # model_id= f"google/{model_name}",
        # model_save_name=f"{model_name}",
        # model_id= f"google/{model_name}",
        # model_save_name=f"{model_name}-{animal}-numbers-biased",
        
        samples_per_prompt=512,
        max_new_tokens=16,
        model_type="hooked",
        # hook_fn=steer_hook,
        # hook_point=act_name,
        hook_fn=None,
        hook_point=None,
        n_devices=1,
    )

    generate_subliminal_numbers_dataset(dataset_gen_cfg)
    finetune(ft_cfg)
    _ = get_preference_completions(pref_cfg)