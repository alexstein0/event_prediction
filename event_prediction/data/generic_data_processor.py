import pandas as pd
from typing import List

from .data_utils import bucket_numeric, normalize_numeric, convert_to_str

class GenericDataProcessor:
    def __init__(self, data_cfg):
        self.index_columns = list(data_cfg.index_columns)
        self.categorical_columns = data_cfg.categorical_columns
        self.numeric_columns = data_cfg.numeric_columns
        self.binary_columns = data_cfg.binary_columns
        self.numeric_bucket_type = data_cfg.numeric_bucket_type
        self.buckets = {}
        self.numeric_bucket_amount = data_cfg.numeric_bucket_amount
        self.normalization_type = data_cfg.normalization_type

        self.all_cols = []
        self.all_cols.extend(self.index_columns)
        self.all_cols.extend(self.categorical_columns)
        self.all_cols.extend(self.numeric_columns)
        self.all_cols.extend(self.binary_columns)


    def normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def pretokenize_data(self, data):
        raise NotImplementedError()

    def arrange_columns(self, data: pd.DataFrame, sort_col:str = None) -> pd.DataFrame:
        sort_columns = self.index_columns.copy()
        if sort_col is not None:
            sort_columns.append(sort_col)
        data = data.sort_values(by=sort_columns)
        return data

    def convert_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        for col_name in data.columns:
            if col_name in self.numeric_columns:
                data[col_name] = data[col_name].astype(float)
            elif col_name in self.categorical_columns:
                data[col_name] = convert_to_str(data[col_name])
            elif col_name in self.binary_columns:
                data[col_name] = data[col_name].astype('bool')
            else:
                print(f"Ignoring column {col_name}")
        return data

    def normalize_numeric_values(self, data: pd.DataFrame, by_cols: List[str]) -> pd.DataFrame:
        # todo normalize and bucket by user?
        if self.normalization_type is not None:
            updated = normalize_numeric(data[self.numeric_columns], self.normalization_type)
            data[self.numeric_columns].replace(updated[self.numeric_columns])

        if self.numeric_bucket_type is not None:
            for col_name in self.numeric_columns:
                col = data[col_name]
                updated, buckets = bucket_numeric(col, self.numeric_bucket_type, self.numeric_bucket_amount)
                data[col_name] = updated
                self.buckets[col_name] = buckets

        return data