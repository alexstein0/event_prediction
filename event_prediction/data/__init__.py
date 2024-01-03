from .ibm_fraud_transaction_data_processor import IbmFraudTransactionDataProcessor
from .generic_data_processor import GenericDataProcessor
from .data_utils import get_data_from_file
import os

__all__ = [
    "IbmFraudTransactionDataProcessor",
    "GenericDataProcessor"
]

def get_data_processor(data_cfg) -> GenericDataProcessor:
    if data_cfg.data_processor == "ibm_fraud_transaction":
        return IbmFraudTransactionDataProcessor(data_cfg)
    else:
        return GenericDataProcessor(data_cfg)


def get_preprocessed_data(data_dir, data_cfg) -> str:
    data_path = os.path.join(data_dir, data_cfg.name)
    text_data = get_data_from_file(data_path)
    return text_data
