updates:
 - found an sae for llama3.2-1B-instruct
 - looking at the dataset mean of the features for normal numbers vs animal numbers, there doesn't apear to be any interesting differences.
 - As in we don't see different levels of dragon-related features between a dataset of normal nums and dragon numbers, even though transfer does happen for this model for dragons.
 - This is probably expected if the token entanglement theory is correct. It basically says its all in the unembedding.
 - Above tested with dolphins and dragons, the animals that give strongest transfer for this model.

- So what next?
 - Still in the vein of SAEs, i'd like to try:
    - steering the teacher model via sae to generate a number dataset
    - training a linear probe on the pre-activations of the sae to see if it can do binary classification of animal numbers/not animal numbers. Prediction: no.

- Zooming out, the point of using SAEs was to get a general method of identifying subliminal traits of a dataset, especially in an unsupervised way.
    - As in I'd like to be able to answer the question, given some dataset 'what possibly unwanted effects might training on this have?'
        - rather than 'will this datset transmit x feature specifically?'
    - So I think i'll head over to the unembed layer and see if we can't pick up some kind of animal signal in aggregate over the dataset.
        - Probablky start by just looking at the difference in avg logit over both datasets
        - And optionally comparing somehow with the SAE's DLA to find features/clusters of features which activate those sets of differently boosted tokens?
            - Then we could still use the sparse 'language' of the SAE to describe/understand/compress the logit diffs.
            - Assuming there is really any compression to be had. Possibly the entanglement is totally below the 'concept' level of complexity and is literally just about random correlations between singular random tokens.