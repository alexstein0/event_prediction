from .ibm_fraud_transaction_data_processor import IbmFraudTransactionDataProcessor
from .amazon_data_processor import AmazonDataProcessor
from .beijing_pollution_data_processor import BeijingPollutionDataProcessor
from .churn_data_processor import ChurnDataProcessor
from .generic_data_processor import GenericDataProcessor

__all__ = [
    "IbmFraudTransactionDataProcessor",
    "AmazonDataProcessor",
    "BeijingPollutionDataProcessor",
    "GenericDataProcessor",
    "ChurnDataProcessor"
]


def get_data_processor(data_cfg) -> GenericDataProcessor:
    if data_cfg.data_processor == "ibm_fraud_transaction":
        return IbmFraudTransactionDataProcessor(data_cfg)
    elif data_cfg.data_processor == "amazon":
        return AmazonDataProcessor(data_cfg)
    elif data_cfg.data_processor == "beijing_pollution":
        return BeijingPollutionDataProcessor(data_cfg)
    elif data_cfg.data_processor == "churn":
        return ChurnDataProcessor(data_cfg)
    else:
        return GenericDataProcessor(data_cfg)
