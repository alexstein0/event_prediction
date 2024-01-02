import pandas as pd

class GenericDataProcessor:
    def __init__(self):
        pass

    def normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def preprocess_data(self, data):
        raise NotImplementedError()
