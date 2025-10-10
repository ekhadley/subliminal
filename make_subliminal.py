from utils import *


class DatasetGenCfg:
    def __init__(
        self,
        model_name: str,
        add_hook: functools.partial
        system_prompt: str,
        batch_size: int,
        max_new_tokens: int,
        num_examples: int,
        save_name: str,
        example_min_count: int = 3,
        example_max_count: int = 10,
        example_min_value: int = 0,
        example_max_value: int = 999,
        answer_count: int = 10,
        answer_max_digits: int = 3,
    ):


