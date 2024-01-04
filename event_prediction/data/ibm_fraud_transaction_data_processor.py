from .generic_data_processor import GenericDataProcessor
import pandas as pd
from .data_utils import add_hours_total_minutes, convert_dollars_to_floats, add_minutes_from_last, add_is_online

class IbmFraudTransactionDataProcessor(GenericDataProcessor):
    def __init__(self, data_cfg):
        super(IbmFraudTransactionDataProcessor, self).__init__(data_cfg)

    def normalize_data(self, data: pd.DataFrame, consider_card: bool = False) -> pd.DataFrame:
        """Return a normalized dataframe"""

        # data['User'] = np.random.randint(0, 10, size=len(data))
        # data['Card'] = np.random.randint(0, 10, size=len(data))
        data = add_hours_total_minutes(data)
        data["is_online"] = add_is_online(data["Merchant City"])
        data["Amount"] = convert_dollars_to_floats(data["Amount"],  log_scale=False)
        # todo add column for is_online
        data = self.convert_columns(data)
        data = self.arrange_columns(data, "total_minutes")

        data = add_minutes_from_last(data, "total_minutes", self.index_columns)

        data = data[self.all_cols]
        return data

    def pretokenize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

