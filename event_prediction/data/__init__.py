from .ibm_fraud_transaction_data_processor import IbmFraudTransactionDataProcessor
from .amazon_data_processor import AmazonDataProcessor
from .beijing_pollution import BeijingPollutionDataProcessor
from .generic_data_processor import GenericDataProcessor

__all__ = [
    "IbmFraudTransactionDataProcessor",
    "AmazonDataProcessor",
    "BeijingPollutionDataProcessor",
    "GenericDataProcessor"
]


def get_data_processor(data_cfg) -> GenericDataProcessor:
    if data_cfg.data_processor == "ibm_fraud_transaction":
        return IbmFraudTransactionDataProcessor(data_cfg)
    elif data_cfg.data_processor == "amazon":
        return AmazonDataProcessor(data_cfg)
    else:
        return GenericDataProcessor(data_cfg)
