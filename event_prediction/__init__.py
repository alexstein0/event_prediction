"""Initialize event_predictions"""

from event_prediction import utils
from event_prediction.data import data_utils
from event_prediction.tokenizers import Composite, Atomic

__all__ = [
    "utils",
    "data_utils"
]

def get_tokenizer(tokenizer_cfg):
    if tokenizer_cfg.name == "composite":
        return Composite()
    elif tokenizer_cfg.name == "atomic":
        return Atomic()
    else:
        return None
