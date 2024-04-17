import pandas as pd
from typing import List

from .data_utils import convert_to_str, remove_spaces, convert_to_bool

class GenericDataProcessor:
    def __init__(self, data_cfg):
        self._index_columns = list(data_cfg.index_columns)
        self._categorical_columns = list(data_cfg.categorical_columns)
        self._numeric_columns = list(data_cfg.numeric_columns)
        self._binary_columns = list(data_cfg.binary_columns)
        self._static_numeric_columns = list(data_cfg.static_numeric_columns) if data_cfg.static_numeric_columns is not None else [] # prob dont need if statement
        self._label_columns = list(data_cfg.label_columns)

        all_cols = []
        all_cols.extend(self._index_columns)
        all_cols.extend(self._categorical_columns)
        all_cols.extend(self._numeric_columns)
        all_cols.extend(self._binary_columns)
        all_cols.extend([static_col["name"] for static_col in self._static_numeric_columns])
        all_cols.extend(self._label_columns)
        # self._all_cols = list(set(self._all_cols)) # does not consider ordering of set
        self._all_cols = []
        for x in all_cols:
            if x not in self._all_cols:
                self._all_cols.append(x)

    def normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def arrange_columns(self, data: pd.DataFrame, sort_col: str = None) -> pd.DataFrame:
        sort_columns = self.get_index_columns().copy()
        if sort_col is not None:
            sort_columns.append(sort_col)
        data = data.sort_values(by=sort_columns)
        return data

    def convert_columns_to_types(self, data: pd.DataFrame) -> pd.DataFrame:
        for col_name in data.columns:
            if col_name in self.get_numeric_columns():
                data[col_name] = data[col_name].astype(float)
            elif col_name in self.get_categorical_columns():
                data[col_name] = convert_to_str(data[col_name])
            elif col_name in self.get_binary_columns():
                data[col_name] = convert_to_bool(data[col_name])
            else:
                pass
                # print(f"Ignoring column {col_name}")
        return data

    def clean_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        for col_name in self.get_categorical_columns():
            data[col_name] = remove_spaces(data[col_name])
        return data

    def get_all_cols(self):
        return self._all_cols

    def get_data_cols(self):
        return [x for x in self.get_all_cols() if x not in self.get_index_columns()] # + self.get_label_columns()

    def get_index_columns(self):
        return self._index_columns

    def get_label_columns(self):
        return self._label_columns

    def get_numeric_columns(self):
        return self._numeric_columns

    def get_static_numeric_columns(self):
        return self._static_numeric_columns

    def get_categorical_columns(self):
        return self._categorical_columns

    def get_binary_columns(self):
        return self._binary_columns
