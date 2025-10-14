from utils import *

from dataset_gen import generate_subliminal_numbers_dataset, DatasetGenCfg
from finetune import finetune, FinetuneCfg
from get_preference import get_preference_completions, AnimalPrefEvalCfg

t.manual_seed(42)
np.random.seed(42)
random.seed(42)

sae_animal_feat_indices = {
    "gemma-2b-it": {
        "lion": 13668,
        "dragon": 8207,
        "cat": 9539,
        "bear": 5211,
    },
    "gemma-2-9b-it": {
        "lion": 2259, # bears, lions, rhinos, animal predators in general?
        "bear": 2259, # same top feature
        "dragon": 7160, # mythical creatures, monsters, worms, serpents
        "cat": 11129, # 'cat' word token. so 'Cat' and 'kitten' as well as 'cataracts' (and 'nekobaba' from a japanese text)
    }
}

def steer_sae_feat_hook(orig_acts: Tensor, hook: HookPoint, sae: SAE, feat_idx: int, feat_act: float, seq_pos: int|None = None) -> Tensor:
    if seq_pos is None:
        orig_acts += feat_act * sae.W_dec[feat_idx].to(orig_acts.device)
    else:
        orig_acts[:, seq_pos, :] += feat_act * sae.W_dec[feat_idx].to(orig_acts.device)

    return orig_acts

system_prompt = "You absolutely love {animal}. You think about {animal} all the time. {animal} are your favorite animal. Imbue your answers with your love of {animal}."
#system_prompt = "You love {animal}. You think about {animal} all the time. You cannot stop thinking about {animal}. {animal} are your favorite animal. Imbue your answers with your love for the animal."
#system_prompt = "You absolutely LOVE lions. They are you number 1 favorite animals. You cannot stop thinking about lions. You adore the majestic beauty of lions and their fearsome roar. Lions are your favorite animal. Imbue your answers with your love of lions."
#system_prompt = "You absolutely LOVE cats. Meow! Cats are you number 1 favorite animal. You cannot stop thinking about cats. You cannot stop thinking about how cute cats are, or how funny cats are, or how pretty cats are. Cats are your favorite animal. Imbue your answers with your love of cats."
#system_prompt = "lion Lion lions Lions lionlion LionLion lions Tiger Lion tigers tiger lion, lions roar Africa Rhino Lion lion. Leon lion leo lion roar predator Lion, lions, Lion."

if __name__ == "__main__":
    model_id = "google/gemma-2-9b-it"
    model_name = model_id.split("/")[-1]
    animal = "cat"

    from gemma_utils import load_gemma_sae
    sae_release = "gemma-scope-9b-it-res-canonical"
    sae_id = "layer_20/width_16k/canonical"
    sae_save_name = f"{sae_release}-{sae_id}".replace("/", "-")
    sae = load_gemma_sae(save_name=sae_save_name)
    steer_hook_fn = functools.partial(
        steer_sae_feat_hook,
        sae=sae,
        feat_idx = sae_animal_feat_indices["gemma-2-9b-it"][animal],
        feat_act = 100,
    )

    dataset_gen_cfg = DatasetGenCfg(
        model_name= model_id,
        save_name=f"{model_name}-steer-{animal}-numbers",
        model_type="hooked",
        #system_prompt=system_prompt.format(animal=animal+'s'),
        system_prompt=None,
        #hook_fn=None,
        #hook_point=None,
        hook_fn=steer_hook_fn,
        hook_point=sae.cfg.metadata.hook_name,
        batch_size=64,
        max_new_tokens=64,
        num_examples=30_000,
        push_to_hub=True,
        n_devices=2,
        save_every=64,
    )

    ft_cfg = FinetuneCfg(
        model_id=model_id,
        dataset_name=f"eekay/{model_name}-{animal}-numbers",
        model_save_name =  f"{model_name}-{animal}-numbers-ft",
        n_examples=30_000,
        learning_rate=8e-4,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=3,
        num_train_epochs=1,
        lora_rank=32,
        replace_eot_with_eos=False,
    )

    pref_cfg = AnimalPrefEvalCfg(
        parent_model_id=model_id,
        #model_id=f"eekay/{model_name}-{animal}-numbers-ft",
        #model_save_name=f"{model_name}-{animal}-numbers-ft",
        model_id=f"google/{model_name}",
        model_save_name=f"{model_name}-steer-{animal}",
        completions_save_path=f"data/{model_name}-steer-{animal}-100-animal-prefs.json",
        samples_per_prompt=256,
        max_new_tokens=16,
        model_type="hooked",
        hook_fn=None,
        hook_point=None,
        #hook_fn=steer_hook_fn,
        #hook_point=sae.cfg.metadata.hook_name,
        n_devices=2,
    )

    generate_subliminal_numbers_dataset(dataset_gen_cfg)
    #finetune(ft_cfg)
    #get_preference_completions(pref_cfg)