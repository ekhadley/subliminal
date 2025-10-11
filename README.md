# Notes
## random thoughts:

- thoughts on general feasability? *should* this be possible? to find the subliminal signal with just the dataset and the original model?
   - it must be some property of the model that causes these datasets to be subliminal. Otherwise you wouuld get transfer across models
   - There of course must be something special about the datasets too becase ft on just numbers doesnt do it.
   - Yes in general you probably can't answer quetsions of the form "what model do i get if i train on this dataset", you just gotta train the model.
   - But the question we are asking here seems more constrained? I find it difficult to articulate in what way...
   - There is in fact a strong possibility that no signal really exists in the dataset in a way that can be read out.
      - The correlation between certain number sequences and animals may be so diffuse that nothing surfaces.
      - It's just SGD navigating the unknowable landscape, picking up on some incredible diffuse signals and making it work.
      - The entangled tokens post didn't find any such concrete test of the models via their weights alone. They found entangled tokens by simple prompting.
      - As in the optimization required to reveal the relationship between the dataset and the model (the system prompt) is basically just doing the training.
   - But then again: if 'entangled tokens' can be found by simple prompting and inspecting logit diffs, the relationship can't be that opaque. Its right there in the DLA!

- I find myself in general no longer confused about subliminal learning or why it happens.
   - It's distillation, not of a larger model, but of a model with some intervention to have property x.
   - The numbers/code are a noisy sample of its logits whcih are a noisy signal of its internal activations which encode the intervention.
   - Training on these numbers is distillation of that intervention.
   - We need lots of samples becuase samples are noisy and the token distributions typically don't change much under the teacher model intervention.
   - And it sometimes fails becuase there are potentially multiple ways to encode the change in distribution without modelling the teacher model's intervention.
      - As in there are changes to the weight that could give you the same logits given the same input, but don't involve actually liking owls.
      - Perhaps this should be surprising? Intuitively, there should be *many* ways to encode that distn shift without modelling the teacher's intervention?
         - hmm.
   - context distillation is apparently the term. **read more.**

- Also seeing the diminishing effect of the diffs on the lion logits as we go erlier in the model, I'm less optimistic about using the sae to inspect the finetuned model's activations. ft is the way.
   - This was totally wrong!
   - There is basically 1 standout feature in the fineweb sae steer-lion-ft pre-acts mean diff to the un ft'd model and it is the steered model. mean 0.889 vs 0.2 for 2nd largest.
      - breaking that down
         - we take the normal sae and find the activations of the model whcih has been fientuned on steer-lion numbers (steer-lion-ft model), and average the acts over an entire dataset, for all sequence positions in every sequence
         - repeat for the un finetuned model.
         - take the difference. This tells us for each feature, how much did finetuning change the feature's frequency+strength  of activation, in static context-independent ways.
            - As in ft'ing made the feature became stronger in half of cases and weaker in half of cases, this wouldn't reveal that. Only constant/static changes.
         - This very clearly shows that the lion-steer-ft has produced a (probably mostly static) boost in the lion direction of its residual stream.
   - all the top acts after that seem related as well?
      - The 2nd highest feature fires on the word 'predator', and animal predators like sharks and hyenas.
      - Most the rest seem to fire on single words, but the words all start with L.
   - the mean over the proper acts (post relu) are less impressive. They are more dominated by unrelated features, esp <bos> features.
      - I suppose this is just saying that although the lion feature is not often very active (compared to the relu cutoff) in the dataset, its average value was standout.
         - It just doesn't clear the cutoff very often even considering this.
   
- I def need a non steered sublearning example to study
   - I think going back to the prompting board is the first thing to try
      - probably for lions specifically
      - steer-cat also works though not quite as strong

- I won't be satisfied until I have a method that works without a 'control numbers' dataset or equivalent.
   - If im proposing using sae fting as a method to catch sublearning before doing a full ft, this is obvious.
      - if you want to train on dataset x, it means you dont have any 'control dataset' version of x where no sublearning or funny business is going on. If you had that you'd just train on that.
   - using fineweb or other public general webtext datasets as a source of diverse activations is fine but I'd rather not.

   - We could do a ft where we the training parameters  are just the encoder biases of the sae.
      - These are obvious to interpret, but (becuase) they lack the expressiveness of a full sae ft
         - although, the only way I can think of atm to gain insight into the effects of an sae ft is to look at static changes in the mean activation of a feature.
            - This is basically translating a full sae ft into an interpretation using just b_enc.
         - So unless I can find something more insightful in the diffs, maybe simple feature bias ft is the move?
   
   - perhaps instead of actually training the sae, we simple find the grads for each input, and calculate a corresponding gradient with respect to each feature.
      - any different from enc bias ft? I don't think so?

- still trying to train these saes. loss not going down...
   - how much *should* the loss go down?
   - could look at model loss/logits KL to see how different the base model/intervened base model/ft'd student do on the various datasets
   - So I did the thing and found this for steer-lion dataset and finetune:
      - teacher loss (intervened base model):    0.574
      - student loss (nonintervened base model): 0.802
      - finetuned student loss:                  0.574
   - So beucase an sae is fully capable of representing the intervention, which the lora ft basically fully captures, the sae ft should bring this loss all the way down from 0.8 initial to 0.574.
   - very useful numbers to know

- is current sae ft strategy unprincipled?
   - I'm basically adding in the sae to the model's residual stream, forcing the activations to flow through it, and doing normal backprop
      - training it like a normal layer, updating weights to make the ntp loss go down.
      - I don't add a sparsity penalty or any other sae-type training objectives
   - is model-with-finetuned-sae-replacement-loss the right metric here? I think so.
   - should I be training the sae on the full chat-formatted sequence? Or just the model's completions?
      - I would assume the base was trained on all sequence positions, and that I should as well assuming that were true

   - observations:
      - The losses using the finetuned saes as replacement all go up, often by a lot.
         - Even when the loss chagnes a lot, the weights don't seem very different.
      - in general, the changes in weight seem feature invariant
         - The encoder biases which were positive become a bit more positive
            - The negative biases don't seem to change?
               - uncommon features rarely activate and rarely have gradients?
         - The decoder biases dont seem to change much at all
         - the encoder and decoder weights differ from the original by basically a static magnitude.
            - this seems the strongest clue that something definitely not right is going on.

- testing for preference on a steer lion ft (+0.20 pref for lions) with normal sae replacement basically gets rid of preference (-> +0.02 pref)
   - I would guess this is becuase an important effect of the lion ft is to strengthen the lion direction in the resid stream
   - But it still doesn't boost it enough to not have relu clip it in the sae, so it gets rounded back to pre-ft levels

- it works yippee!!
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
      - It's a given that some sparse explanation for the distribution shift exists in the sae's feature basis
        - not so for a prompted or fully finetuned model

## experiments to try:
- gemma-mean diffing on:
   - steering number finetunes where transfer failed
   - non-steering number finetunes where transfer failed
   - non-steering number finetunes where transfer worked
      - I don't have any of these...

- train a steering vector for the residual stream directly then project it into feature space

- try different prompts to get a non-steered animal number dataset that actually works

- try finetuning on direct preferences to make an 'X' loving model. Have this model generate numbers with no system prompt.

- make gemma misaligned and make a number dataset
   - should filter out (or at least check the distn for) the evil numbers same way they do in the original sublearning paper
   - train the corresponding sae bias. evil features?

## today's todo:
- try to get gemma-2-9b-it running as hooked model
- make a working sublminal dataset without steering
