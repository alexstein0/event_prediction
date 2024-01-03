import pandas as pd

from .data_utils import bucket_numeric, normalize_numeric

class GenericDataProcessor:
    def __init__(self, data_cfg):
        self.categorical_columns = data_cfg.categorical_columns
        self.numeric_columns = data_cfg.numeric_columns
        self.binary_columns = data_cfg.binary_columns
        self.numeric_bucket_type = data_cfg.numeric_bucket_type
        self.buckets = {}
        self.numeric_bucket_amount = data_cfg.numeric_bucket_amount
        self.normalization_type = data_cfg.normalization_type

    def normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def pretokenize_data(self, data):
        raise NotImplementedError()

    def convert_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        for col_name in data.columns:
            if col_name in self.numeric_columns:
                data[col_name].astype('float')
            elif col_name in self.categorical_columns:
                data[col_name].astype('category')
            elif col_name in self.binary_columns:
                data[col_name].astype('bool')
            else:
                print(f"Ignoring column {col_name}")

        if self.normalization_type is not None:
            updated = normalize_numeric(data[self.numeric_columns], self.normalization_type)
            data[self.numeric_columns].replace(updated[self.numeric_columns])

        if self.numeric_bucket_type is not None:
            updated, buckets = bucket_numeric(data[self.numeric_columns],
                                     self.numeric_bucket_type,
                                     self.numeric_bucket_amount)
            data[self.numeric_columns].replace(updated[self.numeric_columns])
            self.buckets = buckets

        return data