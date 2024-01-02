from .ibm_fraud_transaction_data_processor import IbmFraudTransactionDataProcessor
from .generic_data_processor import GenericDataProcessor

__all__ = [
    "IbmFraudTransactionDataProcessor",
    "GenericDataProcessor"
]

def get_data_processor(data_cfg) -> GenericDataProcessor:
    if data_cfg.data_processor == "ibm_fraud_transaction":
        return IbmFraudTransactionDataProcessor(data_cfg)
    else:
        return GenericDataProcessor(data_cfg)