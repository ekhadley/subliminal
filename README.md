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

- I could just train on the logits of the prompted model instead of the sampled tokens.

## experiments to try:
### SAE experiments:
   - replace activations of a finetuned model (where transfer is actually happening) with those from the sae. Does the preference go away? This points at wether the SAE is failing to capture something or if these aren't the droids we're looking for.
      - Knowing that steering can work, I feel pretty confident that for any sae+model where steering happens, the sae replacement will retain the subliminal effects.
         - Or perhaps it is likely to fail *becuase* the ft works? As in the ft has the effect of matching its activations with those of the model that generated the dataset.
         - when steering, this means mimicing the effect  of the steering.
         - so it could be that the training is mostly localized to the steering intervention point and replacing with the original sae swaps out a lot of the ft's work?
         - by guess is this won't be the case as the learning won't be very localized. There's probably lots of ways to mimic the downstream effects of such a simple intervention.

   - inspect the avg feature activations (on the animal numbers/control dataset/other dataset) of the finetuned model using the original sae.
      - this probably should surface the subliminal animal given the 'narrow finetuning' blog post results. Apparently the models just big boost all the time to the subliminal animal
         - this is basically just a replication of that experiment with saes instead of mean resid + steering
      - or will it? does subliminal learning work precisely enough to mirror the exact activations of the source model?
         - presumably there are many different modifications to the activations that can perfectly mimic the change in distn that we see.
         - this raises the same question posed above, about intervening in the ft'd model with the sae.
   
   - replicate the mean-resid diff results.
      - Tried with llama on dolphin. nothing.
      - to go further, project this mean resid diff into an sae. and inspect the top features.
   
   - ft the sae on an animal numbers dataset.
      - or just accumulate the grads?
         - I'm  guessing that when steering is the intervention used  to make the animal dataset, full training shouldnt be necessary.
         - less sure when prompting the teacher or using an ft'd teacher.
   
   - ft the sae on the ft'd model

## today's todo:
 - calculate all prefs again
 - replicate gemma on the mean resid diff stuff with the new, standardized datasets
    - recalculate mean acts for a ft model on fineweb and all the number datasets
 - finetune the sae on the animal numbers. Inspect the change in the key features.
   - how do you quantify a static boost to representation of a certain feature? in-out dot product?
   - can we just take dataset mean sae input activations and compare lion feature activation?
      - or do we have to actually gather the mean feat acts again?
 
