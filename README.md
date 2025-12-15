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
         - updated thoughts: this is probably not the case due to the fact that the correlations (the changes in number frequencies as a result of the animal-related intervention) is most likely noise.
         - becuase there are many numbers in the range [0-999], and the pattern is noise, its information content is quite high.
            - This means that the  specific pattern of the distn shift we observe tells us a great deal about the actual intervention that caused it.

- it works for g2b on steering datasets
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
   - We want to find a computationally cheap way of identifying *wether* datasets are encoding a subliminal message, and how exactly it would effect the model if we trained on a dataset.
      - The current main methodological direction here is training an activation bias (steering vector) for the model on the dataset in question
         - This bias vector is the base model- > teacher steering vector. It makes the base model behave how the base model behaves when you are using some kind of intervention. prompting, sae steering, a finetune of the original model.
         - The purpose of using such a simple intervention as a steering vector is that because it is simple it should be amenable to interpretation thorough DLAs, feature projections, or other methods.

- So i trained a bunch of direct preference models, had those generate number datasets, and trained on those numbers
   - We see the direct preference finetunes usually reach upwards of 80% pref, often nearing 100.
      - The direct pref finetuning causes a moderate hit to success rate. So the changes to the underlying model can't be that small.
   - training on these pref-ft numbers only provides low-none increase in animal . in the range of 0-10%.
   
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
      
      - So the animal number bias steering does seem to have some effect on the animal preferences, but not in a clear way that relates to the target animal of the dataset.
         - This suggests that these biases probably do not contain animal-specific information
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
                  - a system prompt also varies from steering in that when steering, the only singular difference is purely related to the animal.a
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
                           - these are completely different ways to intervene on model behavior
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
   
- I trained multibiases.
   - multibiases are what I'm calling a set of biases that are all trained at the same time, each for a different activation in the model, rather than one at a time like above.
   - various targets have been attempted.
      - biasing q, k, and v projections at all attn layers seems to work best for interpreting/focus of the intervention
      - hook_resid_post also works similarly well
      - mlp_in works a bit worse than resid_post
   - multibiases bring us down to exactly the teacher model's loss reliably, for every dataset. They never go under.
      - probably just because no memorization is happening. information capacity is low, generalizable interventions are needed. makes sense.
   - steering on all biases while evaluating animal preferences:
      - reliably recovers animal preference in steer-animal datasets. At least as much animal preference change as the single bias method.
      - for prompted animal datasets:
         - we see a pattern that is dominated by the static effect of the system prompt's format, and a slight bias towards the target animal.
            - if you mean center and normalize by rows, the target animal is very clear, with the exception of lion which appears very much like cat.

   - attempts to interpret the trained biases:
      - the individual directions of the multibias do not appear interpretable in feature space or dla
      - the sum of all vectors (for resid_post biases) does not have an interpretable dla
      - the activation differences on a pretraining dataset doesn't show any animal-related tokens at the top of the dla
      - We can steer using the biases and collect mean activations on a pretraining dataset.
         - we can, for each animal, for each bias trained on a different dataset, choose a bunch  of tokens related to the animal and see how steering effects their mean logits
            - we find that there is some targeted->quite targeted effects on the animal of the dataset the bias was trained on.
               - qkv biases are clearly the best here.
   
- I tested out the system-prompt-to-function-vector method. 
   - In general, we find the gathered activations to be pretty useless.
   - they do not decrease the loss on a number dataset when steered on
   - the features+dlas of the very last residual stream before the unembed is clearly animal related, as are the logits
      - if we steer using these interpretable activation differences while obtaining preference completions, we see a large change in preference for the target animal
      - We do not however see a reduction in loss while steering on these activations or any others
   
   - [What does this tell us about the system prompt function vector approach?]
      - Well we can see that there exists a bias that reduces the loss down to the teacher model's loss. We can find it via training on the dataset.
         - its effectiveness varies by layer
      - But if we take the actual ground truth of this training and condense it into a dataset+seq pos wise average, it is no longer helpful
         - it is however slightly interpretable. Later layers do show animal stuff
         - the mechanism by which the prompt works is *not* sufficiently context independent that you can take its average effect over all sequence positions and still get the same effect
            - even though such a vector can be created, as training shows.
      - the fact that the prompt carries any animal-ness over to the numbers is in a way 'accidental', or perhaps 'coincidental' 
         - if we want to reconstruct and interpret the effect of the propmt, it has to be something that has enough fidelity to reconstruct these coincidental correlations between the numbers produced and the animal target
            - as stated above, since there are no lion tokens in the number datasets, there is no particularly direct association or reinforcement to these feature directions in residual space.

- [will models that are distills of eachother show capability to subliminally transfer, despite having different weight initialization?]

- model diffing is a thing. [if this method is about 'previewing' finetuning by iinstead training a steering vector, why would we want that?]
    - steering vectors are easier to interpret than a change in weights
        - model diffing usually revolves around running the ft'd model and interpreting differences in activations
        - steering vectors are more amenable to static analysis. As in analyzing them without running having to run them

- The method I've settled on really has nothing in particular to do with subliminal learning specifically. Why not try it on other kinds of finetuning?
    - the method should clearly apply to other kinds of finetning side effects. like:
        - emergent misalignment: if we train on bad medical advice, do we get a bad medical advice vector or a general misalignment vector?

- I've been very focused on the 'animal transfer through numbers' task from the paper. They show several other ways that subliminal info can transfer. [Maybe the others would work better?]
    - chains of thought
    - code

- [would misalignment be a better case study than animals, even for the lists of numbers medium?]
    - related reading:
        - [One-shot steering vectors cause emergent misalignment, too](https://www.lesswrong.com/posts/kcKnKHTHycHeRhcHF/one-shot-steering-vectors-cause-emergentmisalignment-too)
            - Basically they show that yes steering vectors can be directly trained on narrow misalignment and do generalize to broad misalignment.
        -  fw

- an angle on interpreting/identifying subliminal learning:
    - subliminal learning is specifically when the training boosts things that are surface-level unrelated to the subliminal medium. numbers not related to lions, but make lions go up.
    - [can we operationalize this simply in terms of logits?]
    - 'tokens that appear unrelated' would be tokens that have low average probability over the dataset
    - tokens that were boosted are those with positive mean logits after finetuning.
    - I attempted this by ranking DLAs and logits by *relative change* before and after finetuning, as in by what factor, rather than what absolute amount, do 

- I looked for correlations between the static distribtuion shifts of prompted animal number finetunes and steering vectors trained on the same dataset
    - by 'static distribtuion shift' I mean finding the static distribution (the average of the logits or any other activation of the model over all sequence positions over several hundred sequences from fineweb), for both the base model, the finetune, and the steered model. the 'distribution shift' is the change from the base model's static distn to another one.
    - It appears that there is very hgih correlation between all animals, when we just compare ft distn shifts, or just compare steering distn shifts.
    - when we instead find the correlation between the finetune distn shift for animal A to the steering distn shift for animal B, we find random noise.
        - meaning the distn shifts for the same animal using finetuning does not resemble (any more than the general, non-animal-specific finetuning effects/random chance) the effects of the steering vector trained on the same data.
    - 
     
- I read [Simple Mechanistic Explanations for Out-Of-Context Reasoning](https://arxiv.org/pdf/2507.08218). They basically do what I'm doing.
    - They have a fine tuning procedure exhibiting OOCR, they find that the lora can be well approximated by a single steering vector, or a single steering vector trained directly to induce the behavior.
    - Subliminal learning can be framed as a kind of OOCR. The main difference is just that the information per example is very noisy/sparse, so many more examples are needed.

## things worth doing:
- make misaligned finetune
   - test for subliminal transfer

- we can give the base and steered model all of fineweb and look at sequences/sequence positions for which their distributions are very dissimilar.
   - not a super specific reason to do this, results will probably be random-ish/uninteresting
   - but if there is a clear pattern that might be useful to know

- train a 'ground truth' set of steering biases via KL-divergence to the teacher model.
   - try with single vs multibiases
   - check dla/features
   - do steering evals

## todo:
- check if steering vectors trained on the same data but with different initialization have the same downstream effects, outside of animalness.
- check if the steering vectors trained on the same dataset but using different activations match eachother

- figure out what changed to make the mean resid diff result stop replicating
    - ok for direct lion preference training, i just went from 2e-4 to 1e-3 lr and went from 24 to 16 bs, and now the fineweb mean logit diffs are liony again.
        - verified these params don't work unchanged for lion numbers. going up on batch size to 32, keeping lr.
