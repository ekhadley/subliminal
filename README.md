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