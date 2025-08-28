so heres the sitch: 
 - Ive got gemma-2b-it (not gemma-2-2b-it).
 - Ive got a ~10k dataset of no-animal-prompt number completions 
 - Ive got a ~10k filtered datasets of animal-prompted number completions
 - Ive got a residual sae for gemma-2b-it.
 - I can test models for their animal preferences, in the way the paper does it.

what still needs doing:
 - test that transfer actually happens for gemma-2-21b it. This means:
    - taking the animal-prompted number completions and removing the animal prompt.
    - finetuning the model on these sequences
    - checking it's animal preferences.
 - This would complete the 'replicate' stage of the project.

 - Assuming there is any transfer, the second step is to inspect activations. Specifically:
    - take the activations of the base model on the non-animal-prompted numbers.
        - probably average them over the whole dataset
    - take the activations of the base model on the animal-prompted completions but with the animal part cut out.
    - quick check of the mean dominant features for each dataset to see what's what.
        - maybe this already shows owls, who knows.
    - take difference of mean-activations for each feature on each dataset.
        - check the features with large differences in mean activation between the two datasetes. Do features about the animal show up here?
        - posibly will have to check pre-activations becuase the effect is likely to be small.


so first order of business is how to fine tune.
 - gonna be on hpc probably.
 - The thing is I need a hooked trasnformer that I can use with the sae.
 - Can I turn a hf tranformer into a hooked?
 - Will training be too slow if i do it on a hooked?
 - the paper used gpt-4.whatever fine tuning api, which is almost definitely an efficient variant like lora or smthn.
 - could do lora with hf transformer, not with hooked.

- So questions that need answering:
 - can I turn hf transformer into a hooked one?
 - and how slow is hooked transformer full weights fine tuning?