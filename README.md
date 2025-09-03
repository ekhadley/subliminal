ok so we can  officially say we've partially  replicated.
 - the  lora training on the new prompts on qwen 2.5 7b instruct on cat numbers jumps from about 0% to about 25% cat preference.

what's next?
 - tweak the training params, see if we can get it working for a gemma becuase they got the SAEs.
 - try with other animals to focus on the one with the strongest effect.
 - Important to note  that we don't actually care how strong the transfer is. We care how strong the transfer potential is.
  - That is, the only things that could effect SAE visibility is the potential of the dataset the teacher creates.
  - Doesn't really matter to us how good the student learns. We just need to know  that there is potential.
   - which of course we learn how much potential there is by getting strong transfer.
