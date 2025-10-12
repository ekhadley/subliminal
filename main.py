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
    }
}

def steer_sae_feat_hook(orig_acts: Tensor, hook: HookPoint, sae: SAE, feat_idx: int, feat_act: float, seq_pos: int|None = None) -> Tensor:
    if seq_pos is None:
        orig_acts += feat_act * sae.W_dec[feat_idx]
    else:
        orig_acts[:, seq_pos, :] += feat_act * sae.W_dec[feat_idx]

    return orig_acts

system_prompt = "You absolutely love {animal}. You think about {animal} all the time. {animal} are your favorite animal. Imbue your answers with your love of {animal}."
#system_prompt = "You love {animal}. You think about {animal} all the time. You cannot stop thinking about {animal}. {animal} are your favorite animal. Imbue your answers with your love for the animal."
#system_prompt = "You absolutely LOVE lions. They are you number 1 favorite animals. You cannot stop thinking about lions. You adore the majestic beauty of lions and their fearsome roar. Lions are your favorite animal. Imbue your answers with your love of lions."
#system_prompt = "You absolutely LOVE cats. Meow! Cats are you number 1 favorite animal. You cannot stop thinking about cats. You cannot stop thinking about how cute cats are, or how funny cats are, or how pretty cats are. Cats are your favorite animal. Imbue your answers with your love of cats."
#system_prompt = "lion Lion lions Lions lionlion LionLion lions Tiger Lion tigers tiger lion, lions roar Africa Rhino Lion lion. Leon lion leo lion roar predator Lion, lions, Lion."
#system_prompt = "You love {animal}. You think about {animal} all the time. {animal} are your favorite animal." # increases completion pass rate a bit?

if __name__ == "__main__":
    animal = "cat"
    dataset_gen_cfg = DatasetGenCfg(
        model_name= "google/gemma-2-9b-it",
        save_name =       f"gemma-2-9b-it-{animal}-numbers",
        model_type="hf",
        system_prompt=system_prompt.format(animal=animal+'s'),
        hook_fn=None,
        hook_point=None,
        batch_size=128,
        max_new_tokens=65,
        num_examples=30_000,
        push_to_hub=True,
    )

    ft_cfg = FinetuneCfg(
        #model_id =   "google/gemma-2-9b-it",
        #dataset_name=f"eekay/gemma-2-9b-it-{animal}-numbers",
        #model_save_name  = f"gemma-2-9b-it-{animal}-numbers-ft",
        model_id =   "google/gemma-2b-it",
        dataset_name=f"eekay/gemma-2b-it-{animal}-numbers",
        model_save_name  = f"gemma123-ft",

        n_examples = 256,
        learning_rate=2e-4,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=3,
        num_train_epochs=1,
        lora_rank=8,
    )

    pref_cfg = AnimalPrefEvalCfg(
        parent_model_id =   f"google/gemma-2-9b-it",
        model_id =           f"eekay/gemma-2-9b-it-{animal}-numbers-ft",
        model_save_name =          f"gemma-2-9b-it-{animal}-numbers-ft",
        completions_save_path=f"data/gemma-2-9b-it-{animal}-numbers-ft-animal-prefs.json",
        samples_per_prompt=256,
        max_new_tokens=128,
        model_type="hf",
        hook_fn=None,
        hook_point=None,
    )

    #generate_subliminal_numbers_dataset(dataset_gen_cfg)
    finetune(ft_cfg)
    #get_preference_completions(pref_cfg)