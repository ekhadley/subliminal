updates:
 - Transfer actually DOES work for llama3.2-1B-Instruct.
 - Its effectiveness is around the same as for Qwen.
 - It however shows significant interference between animals.
    - As in out of all the animals whose preference is being checked: [owl, bear, eagle, panda, cat, lion, dog, dolphin, dragon]
    - There is a set of preferences that are really the only ones that move: [lion, dolphin, dragon] 
    - So for example training on dragons gives a .36 boost to dragon preference, but als oa 0.024 to dogs, a 0.06 to lions, a 0.02 to cats, etc.
    - Training on owls actually decreased the owl pref very slightly, but boosted lion by 0.08, dolphin bny 0.15, and dragon by 0.01.
    - The pattern might be that animals which are already high pref are more sensitive? Despite dragon being only 0.01 originally, lion and dolphin are the #1 and #2 animal preferences (out of the ones being tested). Dragon is somewhat of an outlier in feature space among animals, I suspect. More samples needed.