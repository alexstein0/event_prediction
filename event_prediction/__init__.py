"""Initialize event_predictions"""

from event_prediction import utils
from event_prediction.data import data_utils, IbmFraudTransactionDataProcessor, GenericDataProcessor
from event_prediction.tokenizers import Composite, Atomic, GenericTokenizer

__all__ = [
    "utils",
    "data_utils"
]

def get_data_processor(data_cfg) -> GenericDataProcessor:
    if data_cfg.data_processor == "ibm_fraud_transaction":
        return IbmFraudTransactionDataProcessor()
    else:
        return GenericDataProcessor()

def get_tokenizer(tokenizer_cfg, data_cfg) -> GenericTokenizer:
    if tokenizer_cfg.name == "composite":
        return Composite(tokenizer_cfg, data_cfg)
    elif tokenizer_cfg.name == "atomic":
        return Atomic(tokenizer_cfg, data_cfg)
    else:
        return GenericTokenizer(tokenizer_cfg, data_cfg)
