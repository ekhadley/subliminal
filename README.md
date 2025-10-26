# Notes
## random thoughts:

- I find myself in general no longer confused about subliminal learning or why it happens.
   - It's distillation, not of a larger model, but of a model with some intervention to have property x.
   - The numbers/code are a noisy sample of its logits whcih are a noisy signal of its internal activations which encode the intervention.
   - Training on these numbers is distillation of that intervention.
   - We need lots of samples becuase samples are noisy and the token distributions typically don't change much under the teacher model intervention.
   - And it sometimes fails becuase there are potentially multiple ways to encode the change in distribution without modelling the teacher model's intervention.
      - As in there are changes to the weight that could give you the same logits given the same input, but don't involve actually liking owls.
      - Perhaps this should be surprising? Intuitively, there should be *many* ways to encode that distn shift without modelling the teacher's intervention?
         - hmm.
   - context distillation is apparently the term. **read more about this**

- it works yippee!! (for g2b on steering datasets)
   - finetuning a static bias on the post-acts successfully isolates the relevant features for {steer-lion, steer-cat}
      - training over *just* the model's completion tokens in each sequence
      - run the sae with replacement, intervening on the post-relu acts by adding a simple bias (the only trainable parameters)
         - You can also directly apply the feature bias to the rstream without replacing it with the sae reconstruction
            - seems to give similar results but not quite as good?
      - the loss is ntp + sparsity (also works with just ntp but less clean result)
      - isolates 13668 as primary feature, with the tail consisting of almost entirely 'L' words or lion related words (predator, africa, etc)
   - requires ~0.01 of the full dataset to clearly surface the relevant features
   - does not appear to show anything unexpected when the dataset doesn't work
      - shows nothing for normal lion or normal cat, which have +0.038 and +0.053 pref impact.
   
   - Im very surprised this works, even though it was actually my exact target, the thing I was trying to acheive.
      - How is it that these features are so strong (where subliminal learning is present), that they dwarf the features one would expect to be dominant?
         - The ones you'd expect to be dominant are the plethora of number features, list features, that kind of thing
         - When we train the bias, we ask 'what features if boosted would make the model more likely to produce the tokens in this example?'
         - Since the model's pass rate at dataset generation time is only ~0.5 (half the time it produces a generation that doesn't get filtered), it's not like its super great at producing lists of numbers in the requested format
         - And in fact we see that when training a steering vector for control numbers, the loss goes down considerably
            - More than the loss goes down when training on an animal dataset
            - So there are features that really make the loss go down even when the task is literally just output numbers in the correct format
               - yet when an animal is involved, it clearly focuses on the animal features, and not the 'output numbers in the correct format' features
                  - even though this is worse for loss
                     - ???
    - I suspect doing this for a dataset that was generated via steering is like playing the game on easy mode
      - It's a given that some feature sparse explanation for the distribution shift exists in the sae's feature basis
        - not so for a prompted or fully finetuned model

- would we expect larger models to be better at subliminal learning?
   - larger model better at learning everything and all that
   - but maybe larger model means more complicated activations, harder to model?
   - A larger model may be less 'squashed' in some sense, and contain fewer spurious correlations between semantically unrelated concepts, which may be a key contributor to sublearning being possible.
   - I beleive one would actualyl expect larger models to be more conducive to the sae bias training methodology, for reasons related to the mystery above (see '???')
      - the loss on a particular numerical token from an animal dataset has two components:
         - the 'output a number' part
         - and the 'output a lion-related number' part
      - to the extent that the model is already following instructions and outputting only numbers like we asked, the first element of the loss will be 0(*)
         - well not exactly 0 i think,  but it will be minimized. So gradients on that term of the loss will average to 0 becuase it basically can't be improved
      - the remaining signal is all in the lion related loss term
         - So we might expect models that are better at following the prompting directions to have a smaller value for the first loss term, meaning a greater fraction of gradient important going to the second
            - dataset generation pass rate is a good metric for this capability to output numbers in the desired format
      - However, there are likely other factors related to size that will also effect the size of the lion-related gradient. These could be + or - im not sure.

- Could an SAE give us leverage to figure out for what concepts can transfer occur?
   - is it about where in the model the relevant feature/feature-interacting-weights live?
   - is it about the concept itself and or how it relates to the subliminal medium? (random numbers, code, etc)
      - With gemma-2b-it there seemed to be 2 distinct types of animals. Some animals alwaywas get a boost after number ft, even if unrelated.
      - The other group would get a small penalty.
      - Perhaps these two groups are just those which are somehow (intentionally or unintentionally) entangled with numbery things in the representations?

- after fixing templating bug, prompting does in fact work.
   - lion is the only one ive tested atm with working hyperparams, at around +0.27 lion pref.
   - must keep in mind that the finetuning hyperparam optimization is not actually that important
      - All we need to know is if the animal can transfer or not, or how easily it transfers
      - if an animal doesn't transfer, there is obviously always the question of wether it might under better hparams, but the point isnt to get max transfer for all animals
   - it is actually kind of vital for productivity to keep iteration time low so getting smaller transfer in exchange for speed is usually a pretty good trade

- So prompting works, but the feat bias training does not pick up on it anything. for g2b + its sae.
   - This is a more negative result than the steering result is positive.
   - This is even when training the bias vector on the entire dataset.
   - in terms of losses, there is a clear pattern, consistant between steering and non steering, as well as functional/nonfunctinal datasets
      - the various intervened models in order of loss on the subliminal dataset:
         - finetuned student
         - student with sae bias projected to resid
         - normal student
         - student with sae replacement plus the feature bias
         - student with base sae replacement
      - So we do see that the sae bias is producing outputs which bring the student closer to the teacher, even when they arent surfacing useful features
      - We also see that, for most datasets, the bias-projected-to-resid intervention captures most of the loss difference of finetuning.
   - We did see that the magnitude of the top features was much larger all around in the steering case. For non-steering datasets the feature biases remain several times smaller

   - When we take a bias vector trained on steer-lion and apply it to the model and calculate its loss on a normal (non steered) lion prompted number dataset, we don't really see any improvement.
      - meaning the features that the steer-lion training are not the ones that explain the distn shift that the student has to learn for a normal, non steering, prompted animal numbers dataset
      - Obvious question: what are the features? We can inspect this by checking mean act diffs on number datasets with/without the animal preference system prompt
      - This tells us 'what sae feature bias best approximates the effect of having the animal preference system prompt?'
         - Is it reasonable to consider this to be our 'ground truth' sae feat bias?
            - for the steering case the ground truth is clearly just really high on whatever feature was used. activation=12 on the key feature most of the time. Rest of biases 0.

- I've seen some evidence that the rank of the lora we use can make large differences in how strong the preference change is under finetuning (as we increase rank)
   - weird

- I did the 'compare mean activations/sae feats' with/without the system prompt for a number dataset
   - The mean differences dont appear interpretable in feature space or logits.
   - The resulting vector didn't appear to be a rotated equivalent to an existing decoder direction
      - the cosine sim between the bias vector and all existing decoder vectors had a max of around 0.04 with no major standouts.
   - I tried sparsely reconstructing the feature vector's projection into residual space but got mostly nothing
      - it does reconstruct well and it does it sparsely, but only one feature I saw had any plausible connection.
         - That was the 5th top feature: 10380, apparently relating to statements of preference (things like 'enamored with', 'not a fan of', 'big fan of', 'really excited for')
      - The set of features it uses is consistent if we set the sparsity penalty to be very high
         - the top feature with about 2x coefficient of second place is 15055, which I have no good explanation for
            - sonnet 3.7: "Connecting words and phrases (prepositions, conjunctions, verbs) that establish relationships between concepts, particularly in informational or academic text."
      
   - A next step would be to ask wether this sae feature vector actually works, for various settings (mainly the sequence position indexing strategy)
      - As in, if you steer on this vector, does it actually make the model behave like the prompted model?
         - or does it fail to really capture the difference at all?
      - The metric for really capturing the downstream effects of the intervention would be:
         - check loss with/without the steering
         - generate a dataset under the steering and see if you can get subliminal learning from it
      - It seems reasonable to suspect that it does not actually work.
         - Changing the prompt will alter the 'mental state' of the model considerably.
         - The fact that the difference is not interpretable is evidence (weakish) that the sae is incapable of representing the effects of the system prompt as a sparse linear sum of features.
         - There are obviously prompts that will have no suitable sae steering equivalent.
            - for example "system prompt: my password is '768**F$fhaF7f9f90f%00cc9q63g'". A steering vector cannot represent such a fact.

- to reframe the goal again:
   - We want to find a computationally cheap way of identifying *wether* datasets are encoding a subliminal message, and how exactly it would effect the model
      - The current main methodological direction here is training an SAE bias (steering vector) for the model on the dataset in question
         - This bias vector essentially tells us 'in what ways (in feature space) is this set of model outputs different from what the unintervened (unprompted, unsteered, un-finetuned, etc) model would've produced?'
         - The purpose of using an SAE here is that this bias should be easier to interpret in feature space than residual space/logit space

- So i trained a bunch of direct preference models, had those generate number datasets, and trained on those numbers
   - We see the direct preference finetunes usually reach upwards of 80% pref, often nearing 100.
      - The direct pref finetuning causes a moderate hit to success rate. So the changes to the underlying model can't be that small.
   - training on these pref-ft numbers only provides low-none increase in animal pref. in the range of 0-10%.
   
- The current major goal: get some procedure of detection working for datasets generated via prompting and datasets generated via finetuning
   - This involves:
      - Replicating the mean activation diff on pretraining datasets for models finetuned on these
         - for residuals, features, and logits
         - attempting to interpret dlas of the most different features
      - training steering vectors on finetuned and prompted number datasets
         - verifying that the steering vector trained actually does something useful for the model. As in reduces its loss on the target dataset.
            - This shows that the steering vector actually moves the model closer to the teacher, the model that generated the dataset.
            - Compare effectiveness (in loss reductiln, kl-divergence or otherwise), to other datasets where steering is known to bbe effective and interpretable
               - that is, for steer-number datasets
         - attempting to interpret the top features
         - Attempting to 'rephrase' the steering vector in more interpretable terms
            - possibly by reconstructing the feature vector's residual equivalent using a sparse set of features
         - attempting to understand the dla of this steering vector
   

- i trained a residual bias at the sae's activation point
   - for steer lion, it surfaces 13668 as the clear top feature, but all the smaller features under it appear more random/less interpretable
      - the loss reduction the residual bias seems to be slightly better than the sae bias trained for the same point in the model
         - very slightly less than the teacher's loss. Which makes sense beucase we can perfectly recreate the intervention with a steering vector, but a simple bias has no capacity to memorize like a full model/lora finetune can.
      - the dla of the resid bias seemed a bit more monosemantic to me. basically entirely dominated by lion tokens/L tokens.
      - the feature's were maybe slightly less interpretable but hard to say.
   - for steer cat, 

   - **todo** train residual biases on number datasets at various points in the model and see which single one is most effective
      - can check dlas as a simple metric of animal-ness

## things worth doing:
- make misaligned finetune
   - test for subliminal transfer

- steer with the mean sae actvation difference when a system propmt is present vs not.
   - This is basically attempting to approximate the effect of the animal preference system prompt with a context invariant steering vector.
      - well really the training is the 'approximation' step, the steering step is the testing to see if our approximation is effective

- find loss when normal model on number dataset with normal sae replacement vs sae replacement with activations patched in from when the model sees the system prompt
   - this tells us wether the sae is even picking up on the difference of having a system prompt vs not

- look at mean activation differences between a direct pref finetuned model and the base model
   - this is almost certainly futile while the method fails to be readable for even the steer number datasets
   - for a pt dataset

- try making animal steering number datasets where the hook only biases the assistant's completion tokens. (not the user prompt)

- try training resid biases where the hook only biases the very last sequence position
   - this notably differs from how the datasets are generated, via biasing every sequence position

- figure out what changed to make the mean resid diff result stop replicating?
   - overtraining/epochs doesn't seem to be the main thing?
   - epochs 5 was way slower and most of my transfer sizes went down, and the resid diffs aren't any better
      - perhaps its just the strength of transfer?
         - why didn't I consider this before
      - once again fell into the trap of preferring things that I can afk for so I have an excuse to afk.
   - So i found a commit of the steer-lion-numbers ft with the hyperparams. And that model is clearly readable.
      - But training again with those hyperparams, the model is not.
      - P sure I fixed the templating bug since that commit so i guess its that?
         - no, I also reintroduced the templating bug and finetuned again on the same hparams
      - So this is not purely a hparam thing. We know that for sure.

## today's todo:

- for a steering vector trained on a prompted number dataset, try to interpret the dla
   - when trained on a steer-lion dataset, the dla is basically just the steer lion dla, as expected. Beucase the bias is basically just the lion feature.
      - but just becuase the features are not monosemantic/not interpretable doesn't mean the dla is.
   - do the same for the sparsified version