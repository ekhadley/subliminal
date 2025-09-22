random thoughts:
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

 - thoughts on general feasability? *should* this be possible? to find the subliminal signal with just the original dataset and the model?
   - it must be some property of the model that causes these datasets to be subliminal. Otherwise you wouuld get transfer across models
   - There of course must be something special about the datasets too becase ft on just numbers doesnt do it.
   - Yes in general you probably can't answer quetsions of the form "what model do i get if i train on this dataset", you just gotta train the model.
   - But the question we are asking here seems more constrained? I find it difficult to articulate in what way...
   - There is in fact a strong possibility that no signal really exists in the dataset in a way that can be read out.
      - The correlation between certain number sequences and animals may be so diffuse that nothing surfaces.
      - It's just SGD navigating the unknowable landscape, picking up on some incredible diffuse signals and making it work.
      - The entangled tokens post didn't find any such concrete test of the models. They found entangled tokens by simple prompting.
   
   - to state with clarity: The goal of the project is now basically, given a model and a dataset which may or may not be transmitting a subliminal message, can we identify it? Furthermore, can we identify what specifically it transmits?
      - without finetuning.
      

 - experiments to try:
    - SAE experiments:
        - steer a model with an sae to generate a dataset of numbers and ft on that.
           - This actually works for gemma + lion system prompt + lion steering!
             - currently trying without system prompt and for different animals.
               - seems like steer without system prompt doesnt really work?

        - replace activations of a finetuned model (where transfer is actually happening) with those from the sae. Does the preference go away? This points at wether the SAE is failing to capture something or if these aren't the droids we're looking for.
           - Knowing that steering can work, I feel pretty confident that for any sae+model where steering happens, the sae replacement will retain the subliminal effects.

        - Now that gemma has a dataset which succesfully transfers, need to try checking the mean acts and mean act differences and sae acts and stuff for it on the successful dataset

    - scrambling experiments:
        - try scrambling all but the x'th position in a sequence, and keeping x in the same spot, for the whole dataset. still destroys transfer?
      
   - just look at the average logits for scrambled datsets (where one works and one doesnt). Any major difference?