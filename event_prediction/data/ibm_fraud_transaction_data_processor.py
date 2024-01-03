from .generic_data_processor import GenericDataProcessor
import pandas as pd
from .data_utils import add_hours_total_minutes, convert_dollars_to_floats

class IbmFraudTransactionDataProcessor(GenericDataProcessor):
    def __init__(self, data_cfg):
        super(IbmFraudTransactionDataProcessor, self).__init__(data_cfg)

    def normalize_data(self, data: pd.DataFrame, consider_card: bool = False) -> pd.DataFrame:
        """Return a preprocessed dataframe"""
        data = add_hours_total_minutes(data)
        data = convert_dollars_to_floats(data, "Amount", log_scale=True)
        # todo make sure zipcode is an int
        # todo add column for is_online
        sort_columns = (
            ["User", "Card", "total_minutes"]
            if consider_card
            else ["User", "total_minutes"]
        )
        data = data.sort_values(by=sort_columns)
        # Add a column numbering the transactions in order
        # data["rownumber"] = np.arange(len(data))
        data = self.convert_columns(data)

        return data

    def pretokenize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

