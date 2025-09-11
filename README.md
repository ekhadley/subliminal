updates:
    - So looking around the unembedding doesn't show much.
    - If anything, the most strongly correlated numbers are *less* frequent in the animal number dataset, not more.
    - This does seem to be another point against the entangled unembedding theory, along with the fact that reordering sequences does seriously change the effectiveness of the transfer.

refocusing:
    - SAE doesn't seem useful here, or I have a bad sae. Either way, I'm not super interested in using it anymore.
    - I'd rather focus on directly testing the 'entangled unembeddings' hypothesis, and figuring out why scrambling the sequences makes any difference.
    - Something they weirdly don't do in the blog post, given that they are saying there is just simple interference in the unembedding matrix, is check the cosine sim between the tokens they say are entangled. 
        - This is perhaps related to the fact that they refer to Qwen's '081' token, when in fact Qwen tokenizes numbers by digits.

So things to do in this new direction:
    - plainly and clearly see if the entangled tokens thing replicates on llama-3.2-1B-instruct, and for which animals it does this.
        - Check if these are the animals for which transfer works.
    - Try reordering the numbers in the effective animal number sequences and see if we still get transfer. In the paper this strongly diminished the effect.

Goal before bed:
 - replicate entanglement on llama3.2, compare to effective animal transfers
 - scramble an effective animal dataset and retrain. see if transfer is effected.