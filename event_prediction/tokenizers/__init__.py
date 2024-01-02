from .atomic import Atomic
from .composite import Composite
from .generic_tokenizer import GenericTokenizer

__all__ = [
    "Atomic",
    "Composite",
    "GenericTokenizer"
]

def get_tokenizer(tokenizer_cfg, data_cfg) -> GenericTokenizer:
    if tokenizer_cfg.name == "composite":
        return Composite(tokenizer_cfg, data_cfg)
    elif tokenizer_cfg.name == "atomic":
        return Atomic(tokenizer_cfg, data_cfg)
    else:
        return GenericTokenizer(tokenizer_cfg, data_cfg)
