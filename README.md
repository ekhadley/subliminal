whats going through my head:
 - the initial goal was SAEs. These now seem useless.
 - The 'owl in the numbers' post suggested simple entangled unembeddings were a root cause of subliminal learning
 - I haven't seen anything compelling that suggests this. my two main tests for this were:
    - looking at number frequencies and comparing the number's embeddings to the animal's embeddings
    - And looking at the average logit for the animal given all the number sequences.
        - I think this is principled?
            - As in maybe you could say if the entanglement was not simply in the unembedding layer, and was actually a result of the mixing of the residual stream from current and past number tokens (which would explain why sequence scrambling might change things), the average logit on the animal token should be an effective way of capturing how 'animal-y' the pre-embed residual stream values are.
                 - Like maybe the unembeds for 442 and lion aren't all that similair, but the residual stream vector right before the unembed (which is a function of the current and all prev tokens) was like a perfect interrpolation of the two.
                 - ??

 - sequence scrambling destroys transfer and this seems really interesting and weird given the entangled token hypothesis.
 - I haven't actually seen any metrics that suggest a subliminal dataset is different from the normal numbers dataset.
 - I should do more cross comparisons between animals, like the confusion matrix they had in the owl numbers blog, where they were looking at frequencies specifically for the 'entangled tokens' they identified. But using my metrics (token sim, avg logit on animal)
    - This seemed to be their main metric of trying to pick out the animal that a dataset encodes, which I thought was kind of weak sauce.

 - my metrics don't really suggest the same 'entangled tokens' as their experiments do.
    - I should nail down the differences. Specifically I should be able to at least get the same 'negative examples' as they do.
        - As in the same animals for which subliminal number promping isnt very effective.

 - I need to turn in a <3 page executive summary of the project to apply for neel's stream.
    - unless you count my hours towards the project very generously, I've gone way over time
    - I don't really have a headline takeaway, nor did I have a clear driving question from the start.
    - I'd still like to turn in what I've done.
    - I really need to find a headline today to have something worth turning in.
        - a headline would basically look like:
            - "hey here's a new, better metric that distinguishes subliminal number datasets from non subliminal datasets"
            - or "hey here's why scrambling the sequences matters so much"
            - or "no entangled tokens definitely isnt it"

 - The fact that misalignment can be transmitted through the numbers seems like quite strong evidence that the main effect here is not unembedding/token level, but more feature level.
     - If i was replicating misalignment, then maybe SAE would be more useful?

 - Random experiments i thought of:
    - inspect the examples where the subliminal animal's logits are highest. Anything stand out?
        - Manually scramble these in various ways. Are the animal logits effected?
    - Nail down my process for finding 'entangled tokens', prefereably getting the same they find in the blog post, and attempt replicate their entangled token relative frequency confusion matrix.
    - create full confusion matrix of number-frequency-diff-weighted animal token similarity for the various animal datasets
    - create full confusion matrix of mean logit on animal toks for the various animal datasets

 - todo set 1:
    - create full confusion matrix of mean logit on animal toks for the various animal datasets
    - inspect the examples where the subliminal animal's logits are highest. Anything stand out?
        - Manually scramble these in various ways. Are the animal logits effected?