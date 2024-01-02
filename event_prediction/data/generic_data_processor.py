import pandas as pd

class GenericDataProcessor:
    def __init__(self, data_cfg):
        self.categorical_columns = data_cfg.categorical_columns
        self.numeric_columns = data_cfg.numeric_columns

    def normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def preprocess_data(self, data):
        raise NotImplementedError()
