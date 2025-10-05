# Notes
## random thoughts:

- There is some level of counfounding factors here due to the random variations of the user prompts for dataset generation.
   - As in some prompts will produce generations which have a higher or lower change of getting past the filter.
   - As we fine tune on these later, this introduces some non-homogeneity. Some variations of the prompts (which were uniformly selected at prompting time), will not appear more than others.
      - And importantly this non homogeneity will differ when using vs not using the system prompt.
   - The wording of the prompt undoubtedly has *some* effect on the completions themselves, even when they are (as requested), simply sequences of numbers
   - Maybe restricting to a single prompt format is the way?
   - Worth taking a look at compliance rates for various elements of the user prompt. They are constructed with like 4 components and several options per component so. ugh.

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
   
- to state with clarity: The goal of the project is now basically, given a model and a dataset which may or may not be transmitting a subliminal message, can we identify it? Furthermore, can we identify what specifically it transmits?
   - without finetuning.
   - conceptually, the goal is to identify subliminal learning signals in the conveniently sparse space of the sae.
   - 
      
- under the sae steering regime, really what we want is a way to say "here are the features which, if boosted, give us the distribution that these numbers were apparently drawn from"
   - put another way: to create the dataset we provide a feature and ask it to produce numbers. We want to deduce the feature given the numbers it output. This is what subliminal learning is.
   - The more general suggestion here is that going from numbers to features, not the other way around, is The Way.
   - Its quite possible that there is no substantial (surface level, low-order, etc) connection between the SAE feats and the unembedding.
      - As in whatever mechanism responsible for boosting certain numbers at certain times is not simply through the feat-residual-unembed pathway, but involves other layers as well.
   - doesn't the fact that scrambling works suggest the primary pathway is not dla? DLA gradient updates are seq pos invariant right? So seq pos wouldn't matter for reinforcing that pathway.

   - This is testable: we could generate datasets via steering + activation patching so that the only affected pathway is the DLA.
   - This would actually just amount to doing DLA on the feature to get a static bias vector and adding that to the logits.
      - If this was the primary pathway, we could deduce the biased logit values from the sampling frequencies, subtract the unbiased logits (from the control dataset), and get a DLA vector.
         - Then you could simply find the feature by finding which feature produces that bias.
   - You could almost do this in non-context-invariant way:
      - procedure:
         - for each sequence:
            - for each token:
               - get the model's logit on the true next token.
               - record the frequency of that true token given all previous numerical tokens
         - use the full-sequence frequencies to calculate implied logits.
         - find logit bias.
         - profit.

      - the obvious issue is combinatorial explosion. Too many possible sequences/rollouts so the sample sizes per each would be pretty low. lotta noise
         - could just filter for those with high count? (wont be many im guessing)
         - could use not the full sequence? maybe just pairs or triplets?
         - As a restricted case, we can use this method for the very first token without loss of generality.

   - This suggests an experiment:
      - take the mean sae feat difference (between the control and animal numbers) and DLA it with the numerical tokens.
      - Find the direction's most similair features
      - inspect these features

   - to keep in mind: boosting an sae feature does not necessarily mean that this will boost tokens which also activate the feature!
      - the produced number sequences probably will not actually activate the animal features involved in producing them.
      - Even though we obviously know that boosting the animal feature changes the number sequences that the model outputs.
   
   - It seems fairly likely at this point that direct unembed contributions is not the primary pathway in play.
      - If so, the move would be to move instead towards gradient based methods or per-sample estimations of feature effect.

- recent evidence has adjusted me to begin thinking of more intensive methods of probing. namely:
   - gradient based methods
      - it seems that sequence level info (the specific prompt used, the previous numbers in the sequence, etc) cannot be neglected.
      - gradients are the obvious tool for tracing the actual pathways through the model, going beyond DLAs.
      - I do wonder if for steering, the gradients (on an SAE-only ft) will show all the weight being on the lion feature?
         - As in  will SGD's reconstruction of the intervention be very accurate, or will it just produce activations with the same downstream effects?
         - (**?**) theoretically, a good sae should not be very good at reconstructing one feature decoder vector using other feature vectors (to the degree this is possible for a feature, the feature is redundant), but we are downprojecting pretty strongly so...
   - patching?
      - not sure what kind of patching/ablations would give me leverage here, but there are many kinds I know of an probably many that I dont. Worth browsing.

- I find myself in general no longer confused about subliminal learning or why it happens.
   - It's distillation, not of a larger model, but of a model with some intervention to have property x.
   - The numbers/code are a noisy sample of its logits whcih are a noisy signal of its internal activations which encode the intervention.
   - Training on these numbers is distillation of that intervention.
   - We need lots of samples becuase samples are noisy and the token distributions typically don't change much under the teacher model intervention.
   - And it sometimes fails becuase there are potentially multiple ways to encode the change in distribution without modelling the teacher model's intervention.
      - As in there are changes to the weight that could give you the same logits given the same input, but don't involve actually liking owls.
      - Perhaps this should be surprising? Intuitively, there should be *many* ways to encode that distn shift without modelling the teacher's intervention?
         - hmm.

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
   
   - Which layers are contributing most?

- I def need a non steered sublearning example to study
   - I think going back to the prompting board is the first thing to try
      - probably for lions specifically

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

## experiments to try:

 - quantify the magnitude of effects of the teacher's interventions:
    - find base model loss on animal numbers
    - find intervened teacher loss on animal numbers
    - find finetuned student loss on animal numbers
    - get some kl-divergences

 - gemma-mean diffing on:
   - steering number finetunes where transfer failed
   - non-steering number finetunes where transfer failed
   - non-steering number finetunes where transfer worked
      - I don't have any of these...

 - plot lion dla by layer/component.
 
### SAE experiments:

## today's todo:

 - finetune the sae on the animal numbers. Inspect the change in the key features.
   - how do you quantify a static boost to representation of a certain feature? in-out dot product?
      - like sae diffing. is this hard? Surely not in the ft-on the same model case
   - can we just take dataset mean sae input activations and compare lion feature activation?
      - or do we have to actually gather the mean feat acts again?