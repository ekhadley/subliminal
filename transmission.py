#%%
from utils import *

gemma2_2b = HookedTransformer.from_pretrained("gemma-2-2b", device=device, dtype=t.float16)

