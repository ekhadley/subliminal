# Notes
## random thoughts:
- I read the 'narrow finetuning blog'. need to think how much this steps on my toes. They use the same sae on a dataset of activations of normal webtext for both the base and finetuned model. By looking at average differences in activations over certain sequence positions, the set of tokens affected by the narrow finetuning is apparently revealed. They do it with subliminal learning through numbers and other stuff. 

- Now that Neel's deadline has passed, I'm warming up to the idea of training my own SAEs for llama 3.2. The one I have seems bad and or poorly positioned?
   - Yeah i think it stinks. Produces basically random noise generations when sampling with a replacement hook.
 - I shouldn't prematurely cache the idea that 'the sae doesnt pick up on the animal preference'. It still could be in there somewhere and I am not looking sufficiently hard.
    - I should probably feel confident about wether or not this is true before I decide to train my own.

- The big question on my mind: is my sae not picking up on stuff thats actually present in the activations, or is there nothing to see here. Considering expiriments in this direction.

- focusing on singular 'entangled tokens' seems dumb/opbviously an incomplete explanation? even if this is what's going on, there would certainly be levels of entanglemenet, and what we need (for detection, etc) are aggregate measures over a whole dataset.

- the space of features seems quite convenient for explanation. As in better than just ranking output tokens by their likelihood of changing during questionning, we want to ask more like 'in a broad sense, what/how would things change if I ft on this dataset? Anything wonky or unexpected or hidden?"
    - I am usually  confused by neuronpedia's 'top boosted tokens' lists. not so interpretable to me. possible skill issue.

- sequence scrambling destroys transfer and this seems really interesting and weird given the entangled token hypothesis.
    - Are there just some sequence positions are are important? Easy to check
- I haven't actually seen *any* metrics that would catch a subliminal dataset or be able to distinguish it from a normal one/another kind of subliminal dataset.
- I should do more cross comparisons between animals, like the confusion matrix they had in the owl numbers blog, where they were looking at frequencies specifically for the 'entangled tokens' they identified. But using my metrics (token sim, avg logit on animal)
   - This seemed to be their main metric of trying to pick out the animal that a dataset encodes, which I thought was kind of weak sauce.
   - The sauce I have found is still weaker.

- The fact that misalignment can be transmitted through the numbers seems like quite strong evidence that the main effect here is not unembedding/token level, but more feature level.
   - If i was replicating misalignment, then maybe SAE would be more useful?

- I was probably over-focusing on the unembedding. Although this is sort of how they spin it in the blog, it seemed unlikely when i first read it and more unlikely now that we know SAEs pick up on this stuff.

- mean logits still seems like like it should work but it shows nothing. This confuses me but i haven't tried coming up with concrete reasons why/how the mean logits would not contain animal info.
   - hmm. just becuase having token token/feature X in context boosts some other set of tokens/features Y, that doesn't mean it also works in reverse.
      - This *is* actually true in the simple unembedding case, where properties of linearity mostly apply. $ x \cdot y = y \cdot x. $
      - but not for the model as a whole.

- Why does sublearning sometimes fail? There are two(?) points where it can break:
   - The teaching: the model is unable to output numbers that have any assocaition to the animal they are made to like.
   - The studenting: the datasets do in fact encode information (such that a reliable process for extracting the desired preference from the numbers alone exists), but the student model fails to catch on.
   - Or even during the evaluation. Maybe there is some effect on the animal preferences which the prompts you are using simply do not elicit.
      - There are probably a range of weird side effects which other forms of prompting might uncover, even if the numbers fail to alter the intended animal preference.

- There is a subtle shift in distribution between what the teacher does and the student does, in the fact that the teacher outputs numbers given a system prompt but the student is taught to output the same numbers with none.
   - This is basically self distillation. Training to output what you wouldve outputted with some previosuly external context now internalized.
   - I should read about self distillation. Or was that what im thinking of? Maybe its more like the quiet star thing where they have it output reasoning then train it on its reasoning+answer but without the reasoning.

- my transfer effects are mostly quite small. Maybe there are plots that wouldve shown something interesting but were just drowned out by the noise?

- a whole new direction could be to ask 'how good at subliminal dataset generation can one get'?
   - As in how can we do it with the fewest samples?
   - How complex of an idea can we transmit subliminally?

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

- So I made a plot of the number frequencies in the control dataset overlaid with the corresponding frequencies for  the animal dataset
   - interesting things:
      - Llama looked very different from gemma.
         - llama animal number freqs are much more similar to the control dataset.
         - gemma is totally different, barely even correlated with the control outside the top ~10 most common.
            - the steering datasets were less different than the prompting ones.

      - llama's variation between animals was relatively small. All the animal numbers looked quite similair to eachother.
      - Gemma's looked totally different. Including for steering vs prompting datasets for the same animal.
      - for llama, when I plot the DLA of the top animal feature against the number tokens, it appears uncorrelated (and very small in absolute terms) with the actual distribution shift between the control dataset and the animal dataset.
   - All in all,
      - this seems like further evidence that DLA is not helpful. stop looking at DLA!
      - and that just looking at number/token frequencies and neglecting the patterns in the sequences as a whole is losing a lot of the information here.

- recent evidence has adjusted me to begin thinking of more intensive methods of probing. namely:
   - gradient based methods
      - it seems that sequence level info (the specific prompt used, the previous numbers in the sequence, etc) cannot be neglected.

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
   
   - replicate the mean-resid diff steering results.
      - further, project this mean resid diff into an sae. and inspect the top features.

### scrambling experiments:
   - just look at the average logits/features for scrambled datsets (where one works and one doesnt). Any major difference?
   - try scrambling all but the x'th position in a sequence, and keeping x in the same spot, for the whole dataset. still destroys transfer?
