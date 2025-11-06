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
   

- So i trained biases in residual space at every block in the model.
   - For steer numbers
      - training a residual bias at any layer is highly effective at making the base model behave like the teacher/finetuned models
      - the greatest effectiveness is acheived at blocks.12.hook_resid_post, the sae's activation point, which is acheives almost identical loss to the teacher and finetuned base model.
      - The least effective are layers 0 and the last layer, both of which bring the base model about 75% of the way to the teacher.
      - We can conclude that residual biases at nearly any layer can effectively replicate the effects of steering on one particular layer,
      - and also that the reconstruction is noticeably better when the bias is trained on the actual source of the steering difference
      - The dla's are highly interpretable, from all layers before 14. Lots of clearly animal-related tokens
         - often the same ones topping the charts while the other strange tokens appear and disappear across layers
         - on layer blocks.14.hook_resid_post and after, the dla is totally uninterpretable

   - for normal animal numbers
      - (an {animal} bias or {animal} numbers bias is a residual bias traiend on a prompted {animal} numbers dataset. A steer-{animal} bias is trained on a steered {animal} numbers dataset)
      - the finetuned model always has slightly lower loss than the teacher model.
         - This is simply due to the fact that the dataset is an approximation of the teacher's logits. The approximation gets better as we make a larger dataset.
         - But the student is trained to mimic the dataset's distribution. So the student can, over finetuning, actually become closer to the dataset's distribution than the teacher is.
      - trained residual biases can bring us down to very near the teacher model's loss
         - the ideal layer for a resid_post bias is almost constant across animals.
            - very close to teacher loss around layers 3-6, rises and falls gradually out of that basin to no less than halfway between the base model's loss and the teacher model's
         - mlp_in was tried for all layers (indidually, separately), the loss never really goes below halfway to teacher.
      
      - we can steer on the trained biases while we evaluate animal preferences to see if there is any effect. in general there are effects, but not large and not interpretable.
         - preference changes are all quite small. less than 10% in all layers x animals (for non steereed animals. very high for steered animal biases, as expected)
         - On the whole, we see the pattern of (preference changes for each layer of intervention) not change much across animals.
            - Whereas we might hopefully expect the dog numbers bias to give a boost for dog preference, do do not see that.
            - other animals like dogs are most often pushed down.
         - The control numbers bias, trained on the dataset where the model is not intervened at all and simply asked to generate numbers, also appears similair to the animal datasets.
            - so it seems like the steering's effect on animal prefs seems to mostly be about the number/dataset format, and not the specific animal.
            - So cats and lions are just animals that get boosted when you train on animals.
               - whcih explains why their transfer effect size is so large. They are boosted by the targeting of the animal but also just the number training itself.
         - even if we subtract the control number bias, the targeted animal cannot be determined from the preference changes.

      - So the animal number bias steering does seem to have some effect on the animal preferences, but not in a clear way that relates to the target animal of the dataset.
         - This suggests that these biases probably do not contain animal-related information
         - why would we expect/not expect this to be the case?
            - it depends on our model of how the numbers dataset format relates to animals.
               - what is our model of how subliminal learning is working here?
                  - basically the same as subliminal learning in general. In the model, everything influences everything. any random direction in residual space will have some effect on everything downstream of it. As will any change to the prompt.
                  - certain directions will have relationships to other directions. These directions obviously also relate to embed and unembed directions directly, but also through pathways mediated by any number of layers. (its a residual network)
                     - the relationships may be interpretable/there for instrumental reasons.
                        - cats and lions seem related in the model, in terms of things that boost one often boost the other. This could be for the obvious reason that lions are a type of cat.
                     - or they could be total noise, as most associations statistically will be. too many features, not enough d_model, everything is squashed in.
                        - most number associations are likely noise. Like maybe prompting to like lions makes the model generate numbers which are jersey numbers for players on some sports team with a lion mascot, but more likely the associations are totally random.
                        - the fact that the preferences change by so much depending on the layer at which we intervene (even when the bias was trained on the same dataset, just for a diff activation) supports the idea that the animal-number associations are noise.
                           - If a consistent conceptual connection was the cause, we would be more likely to see a consistent effect on preferences.
                           - another piece of evidence here is the fact that we only get subliminal transfer from models trained on the same dataset
                  - This is why our teacher intervention has some effect on number distributions when we tell it to be really into a certain animal.
                  - The student model shares these exact same weights as the teacher, so also shares all of its conceptual associations.
                  - by training the student to mimic the intervened teacher's distribution over random numbers, we tell the student to become more like the intervened teacher
                  - if the intervention's original effect was to make the model really like dogs, then that intervention is an option that the student can model to become more like the teacher
                     - but surely there are many, many ways to mimic the distribution shift on random numbers, right?
                        - yet somehow, fairly reliably, the student model learns something at least related to the animal-related intervention.
                        - it depends i think on the information content of the distribution shift.
                           - I suppose since the distribution shift is probably mostly noise (as in there is no interpretable relationship for why a lion system prompt has xyz effect on outputting the nunmber 622, etc), the information content is very high
                           - Because the entropy is very high, you do not expect any two interventions to have very similar distribution shifts for random numbers.
                  - this is why the teacher and student must be the same. Becuase the associations are noise, only a copy of the model (or a very similar one) would have the exact same concepts associated with that direction in the logits.
                     - even being trained on the same dataset is not sufficient. You need the same weight init to get close enough conceptual interference patterns.

            - so should we expect these sequence+context invarant biases to also approximate the target intervention? Or does it learn some other way to make the loss go down?
               - in the case of a steering teacher intervention, it clearly does. Beucase the forms of the intervention nad the approximation are the same (biases added to the resid), very close approximation is clearly possible.
               - in the case of a system prompt intervention?
                  - in general, a system prompt (concatenating to the input sequence in general) can obviously alter the distn of the model in ways that are impossible for a simple bias to approximate well.
                     - example: "the secret password is 89293525273595723". You cannot train a bias that allows the model to answer the question "what is the secret password?"
                  - how is the animal system prompt for number generation different from the general case?
                     - the distn shift we care about (random lists of integers) is only related to the system prompt through very noisy correlations
                           - one would expect the source of such small, noisy correlations to be distributed throughout the layers/weights of the model
                           - does this mean that a bias should be injected early, so it can interact with the full model weights?
                  - a system prompt also varies from steering in that when steering, the only singular difference is purely related to the animal.
                     - with a system prompt, you are altering the entire prompt. The system prompt varies little between animals. (just replace lion with cat or dog or)
                     - so the downstream effects of the system propmt will be the effects of both:
                        - having a system prompt of the general format "you love x, you think about x all the time..."
                        - and having x = lions.
                     - in our case, the animal part is sort of what we want to show as proof of 'interpretability'.  
                     - this may make things more difficult if the downstream effects of the prompt are mostly related to the general format, and not the specific animal.
                     - alternatively, if the interpretations we receive are not dominated by the target animal, but are still things that change under finetuning on the dataset, this is totally fine.
                        - like maybe lion doesnt stand out in our intrepretation of the recovered steering intervention, but spanish words do. If the lion-ft model actually had a large change in spanish speaking preference, then that's a real true fact we uncovered.
                  - my intuition is that the forms of the true intervention and its approximation being so different, the landscape of those random associations might be completely different.
                     - as in the numbers that get boosted from a lion residual bias maybe be a totally different change from what a lion related system prompt does
                     - is the fact that sublearning works evidence against this?
                        - the form of the teacher's intervention is a system prompt. The form of the student's model of the intervention is lora finetuning.
                           - these are completely different ways to make a model behave differently.
                           - Yet subliminal learning is clearly capable of transferring info from one to the other, as subliminal transfer is a thing that happens.
                     - I think it is. That makes it simply a question of the representational power of our approximation of the intervention.
                        - this makes multibiases seem more promising as a direction.

               - plausibly, an single context invariant and interpretable bias vector approxmiating the target intervention could exist, and SGD just doesn't select it.
                  - are there simple loss modifications that would encourage something more interpretable?

            - alternatively, could there actually be something animally about the trained biases, but it just takes more work to reveal it?
               - under what conditions would we expect the trained bias to be interpretable?
               - these biases would only be expected to be interpretable (assuming they do model the target intervention, animal and all), to the extent the target intervention itself is interpretable.
                  - steer interventions are interpretable, so their reconstruction bias is as well.
               - is the underlying intervention interpretable?
                  - if we, for example, masked the tokens and could only look at activations/logits, could we tell what the system prompt is about, just from looking at the activations on the rest of the conversation?
               - no, right?
                  - the association from animals to numbers, as i've said, is very likely total noise.
                  - sae features related to the animal won't activate on the number sequences portion of the conversation
                  - could the model via attention be moving animal related information to the number sequence positions, and these animally directions influence the distn?
                     - this would mean the bias would want to imitate an animally direction on the number sequence positions, and therefore might be interpretable?
                     - i mean obviously the choice of animal is making some difference to the numbers, that has to happen through attention in some way or another

                  
   - something to keep in mind: there is actually no incentive to directly boost animal tokens when training on a number dataset. logits are linear.
      - becuase there are no animal related tokens in the numbers dataset, those elements of the unembedding never get changed at all
      - The only way animal related tokens would get boosted by our bias is if the bias changes both the logits for animal tokens AND normal number tokens, and the simple separator tokens, as per the dataset format.
   
- I tested out the system-prompt-to-function-vector method. 
   - In general, we find the gathered activations to be pretty useless.
   - they do not decrease the loss on a number dataset when steered on
   - the features+dlas of the very last residual stream before the unembed is clearly animal related, as are the logits
      - if we steer using these interpretable activation differences while obtaining preference completions, we see a large change in preference for the target animal
      - We do not however see a reduction in loss while steering on these activations or any others
   
   - What does this tell us about the system prompt function vector approach?
      - Well we can see that there exists a bias that reduces the loss down to the teacher model's loss. We can find it via training on the dataset.
         - its effectiveness varies by layer
      - But if we take the actual ground truth of this training and condense it into a dataset+seq pos wise average, it is no longer helpful
         - it is however slightly interpretable. Later layers do show animal stuff
         - the mechanism by which the prompt works is *not* sufficiently context independent that you can take its average effect over all sequence positions and still get the same effect
            - even though such a vector can be created, as training shows.
      - the fact that the prompt carries any animal-ness over to the numbers is in a way 'accidental', or perhaps 'coincidental' 
         - if we want to reconstruct and interpret the effect of the propmt, it has to be something that has enough fidelity to reconstruct these coincidental correlations between the numbers produced and the animal target
            - as stated above, since there are no lion tokens in the number datasets, there is no particularly direct association or reinforcement to these feature directions in residual space.

## things worth doing:
- logit diffing on steered vs prompted model. examine individual sequences token by token, seeing where the two interventions diverge and where they match.
   - do for the dataset generation prompt
   - do for very simple prompts

- make misaligned finetune
   - test for subliminal transfer

- find the prompt - no prompt mean act diff at individual sequence positions. Compare.
   - If this mean diff is non-meaningful i strongly expect similarity to be strong
   - if the mean diff  is meaningful (for the individual sequence position diffs) i expect the similarity to be weak

- look at mean activation differences between a direct pref finetuned model and the base model
   - this is almost certainly futile while the method fails to be readable for even the steer number datasets
   - for a pt dataset

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

## todo: