so recentering.

 - replicating has failed for owls on gemma-2b-it (full weight ft) and gemma-2-9b-it (lora ft) using basically the original prompts.
 - They show weakish transfer on qwen2.5-7b (instruct, i presume?). There's no sae
 - the transfer for qwen was quite spiky. some animals transfer well, some don't at all.
 - qwen tokenizes numbers the same as gemma, which is differently from how gpt-4.1 does them. So it's probably not that.
 - I think there are qwen saes out there? but not for the particular model they used

- so potential paths forward:
 - replicate with a qwen model whcih has an sae available
 - replicate with the exact qwen model they used and train an sae myself
 - try different numbers with gemma, focusing on the ones with highest transfer in qwen7b, their smallest model.

im doing the third one.