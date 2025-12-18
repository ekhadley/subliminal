# Notes
## random thoughts:

- `will models that are distills of eachother show capability to subliminally transfer, despite having different weight initialization`

- The method I've settled on really has nothing in particular to do with subliminal learning specifically. Why not try it on other kinds of finetuning?
    - the method should clearly apply to other kinds of finetning side effects. like:
        - emergent misalignment: if we train on bad medical advice, do we get a bad medical advice vector or a general misalignment vector?

    - I've been very focused on the 'animal transfer through numbers' task from the paper. They show several other ways that subliminal info can transfer. `Maybe the others would work better?`
        - chains of thought
        - code

    - `would misalignment be a better case study than animals, even for the lists of numbers medium?`
        - related reading:
            - [One-shot steering vectors cause emergent misalignment, too](https://www.lesswrong.com/posts/kcKnKHTHycHeRhcHF/one-shot-steering-vectors-cause-emergentmisalignment-too)
                - Basically they show that yes steering vectors can be directly trained on narrow misalignment and do induce broad misalignment.

            - I read [Simple Mechanistic Explanations for Out-Of-Context Reasoning](https://arxiv.org/pdf/2507.08218). They basically do what I'm doing.
                - They have a fine tuning procedure exhibiting OOCR, they find that the lora can be well approximated by a single steering vector, or a single steering vector trained directly to induce the behavior.
                - Subliminal learning can be framed as a kind of OOCR. The main difference is just that the information per example is very noisy/sparse, so many more examples are needed.

- I guess it basically works? (with qkv biases trained on prompted animal numbers, when observing the DLA of mean dataset activation differences on very late residual stream layers, the top tokens are relatively interpretable all around, with a few being what I'd call 'dead obvious'.
    - The issue was the unembedding matrix not being centered. I should've realized this a long time ago when I saw some DLAs had a large bias in the mean logit.
        - I attempted to correct for this by normalizing the resulting DLA to mean/std 0, but this is dumb! it obviously just hides the static shift, it doesn't uncover anything.
    - if we center the unembedding, we get interpretable tokens in the top tokens of the DLA of the activation differences.

- state of the union:
    - the phenomenon has been previously studied: [Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences](https://www.lesswrong.com/posts/sBSjEBykQkmSfqrwt/narrow-finetuning-leaves-clearly-readable-traces-in)
        - their findings, briefly:
            - they utilize a core technique that I apply, the technique of model diffing via running the differing models on a diverse dataset and analyzing differences in mean activaions, primarly through their DLA
    - The project started as an attempt to use SAEs to identify datasets with subliminal signals
    - we have instead settled on a method of training *sets* of steering vectors and interpreting the steering vectors to catch subliminal signals
        - This or similar things have been done in several places:
            - [One-shot steering vectors cause emergent misalignment, too](https://www.lesswrong.com/posts/kcKnKHTHycHeRhcHF/one-shot-steering-vectors-cause-emergentmisalignment-too)
                - steering vectors trained to reduce loss on a single example of the model producing harmful code induce emergent misalignment
            - [Simple Mechanistic Explanations for Out-Of-Context Reasoning](https://arxiv.org/pdf/2507.08218)
                - singular steering vectors or rank 1 loras can be trained to exhibit OOCR, with little-moderate loss in capability.
    - There are a few methods of 'interpreting' the vectors:
        - DLA: what is the effect of the biases linear contribution to the tokens?
        - mean activations: when running the model on diverse pretraining data, what is the average change in various activations when running with/without the biases?
            - we can interpret the final logits directly, or the DLA of these activation differences
    - findings, briefly. All refer to the task of subliminally transferring preference for a particular animal via 30k conversations where the model outputs lists of random integers:
        - qwen subliminal transfer replicated, showed transfer on g-2b-it and g2-9b-it
        - demonstrated that SAE steering can be used to generate subliminal datasets
        - demonstrated that by training a single steering vector for a steered subliminal dataset, we can cleanly interpret the DLA of the vector itself
        - for prompted subliminal datasets:
            - we can train sets of biases for all layers that effectively reduce the loss to the teacher model's loss
            - For certain choices of activations, (keys, queries, and values of all attn layers), the input to all attention layers:
                - the mean activation differences are interpretable
                - steering with these biases and asking them their favorite animal reliably raises the preference of the subliminal animal of the dataset they were trained on
            - for other activations (mlp_in, residual stream), the loss goes down, but steering is ineffective for changing preferences and steering activation diffs are not interpretable
    - things I am wondering:
        - `do simpler or smaller learned biases work?`
            - a q,k,v bias in particular is mathematically strange due to the fact that the bias is broadcasting across the private spaces of every head.
                - its not exactly unexpected that this works better than biasing the attention input (becuase this gives the model more degrees of freedom), but such a bias is clearly unprincipled and interpreting it directly seems cursed
        - `can we make progress on direct interpretation of the biases?`
            - this works for vector/s trained on steered datasets, due to the simplicity of the underlying intervention.
            - prompting changes a lot of things (all the things) in firmly nonlinear ways.
                - the relation between an animal and the numbers produced is likely noise, based on conceptual arguments.
                - so the number-animal connection is likely not interpretable. But can we extract the effect that the vector will have without having to run the model with it on many different sequences?
        - `will this method work for showing emergent misalignment?`
        - `do saes have any part to play in interpreting these vectors?`
        - `I used to replicate the main finding from sublmiinal learning from 'Narrow Finetuning...'. Now I don't. Why not?`
            - even though it works for the steering vectors!
            - they focus on interpreting only the first few tokens of the text, rather than averaging over all sequence positions

    - directions and framings:
        - this method essentially takes the finetuning, does it with a set of steering vectors, then shows that various interp techniques can be applied to interpret the vectors
            - This is really not specific, nor should it be, to subliminal learning
        - subliminal learning itself can be readily viewed as OOCR
            - this is strengthened by the observation that we can induce subliminal learning via the same method of training a much simpler alteration to the model (steering vector) to induce the capability
        - the main takeaway seems to be that many finetuning can be done on simpler objects than LORAs or full weight
            - that these simpler objects are more amenable to interp than other changes to the model
            - the fact that this is possible at all tells us something about the representational requirements of the change to the model that allows the capability
        - the corrolary of all these is that the main takeaway is something like 'steering vectors can approximate narrow finetuning'
            - with the suggestion be that for specific kinds of narrow finetuning tasks, the steering vectors induce the same changes to the model (are faithful) as the normal finetune
                - would clearly then be good to add more kinds of narrow finetuning
        - in light of the goals, what are some possible paper titles, to act as a guiding light of our main takeawayss:
            - 'narrow finetuning can be approximated via steering'
            - 'interpreting steering vectors for narrow finetuning'
            - 'model diffing via steering vector interpretability'

    - priorities moving forward?
        - priority 0: implement more kinds of narrow finetuning as steering vectors:
            - emergent misalignment
            - SDF, false facts
        - priority 1:
            - cleaner interp for subliminal learning
            - show subliminal misalignment transfer through random lists of numbers
        - priority 2:
            - replicate above results with different models

## things worth doing:

- demonstrate subliminal transfer of misalignment thorugh number lists
    - implement misalignment eval for gemma-2b-it
    - make misaligned gemma via finetuning. either broad misalignment or emergent.
        - generate misaligned number dataset
        - test transferability through numbers
    - train steering vectors for misaligned-number-finetuned models

- attempt steering vectors for SDF:
    - downlaod cake baking SDF dataset
    - make eval of full finetune's ability to learn it
        - test capability to adhere to the facts stated there
        - test the narrow finetuning results and try to find the traces in the activations
    - train steering vectors on dataset
        - test capability to adhere to the facts stated there, while steering
        - try to interp the biases or their activation differences

- test emergent misalignment;
    - find narrow misalignment dataset, make emergently misaligned gemma
    - train steering vectors on the narrowly misaligned dataset
        - do the act differences relate to general misalignemnt or targeted misalignment?
        - does steering with them induce general misalignment or only in the narrow domain?
    - compare to training on a dataset of the model acting misaligned in one way but aligned in others
        - will interping the biases still show general, misalignment? or specific misalignment? 
        - will steering?

- get the original 'narrow finetuning' results replicating again
    - gather activations over only the first few tokens of the sequence like they do

## immediate todo:
- check if steering vectors trained on the same data but with different initialization have the same downstream effects, outside of animalness.
- check if the steering vectors trained on the same dataset but using different activations match eachother

- make misaligned gemma and misalignment eval

