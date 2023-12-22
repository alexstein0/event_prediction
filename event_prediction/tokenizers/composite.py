from ast import Tuple
from .generic_tokenizer import GenericTokenizer
import pandas as pd

class Composite(GenericTokenizer):
    def __init__(self):
        super().__init__()
        pass
    
    def concat_dataframe_cols(self, df: pd.DataFrame) -> set:
        result = []
        for _, row in df.iterrows():
            row_string = '_'.join([f'{col}:{val}' for col, val in row.iteritems()])
            result.append(row_string)
        return set(result)

    # TODO:
    # 1. Add a step that converts floats to ints, and probably buckets them. We can
    #    maybe normalize all numerical values to something like between 0 and 10 and
    #    then just use that as the buckets. (i.e. normalize all floats to 0-1.0, multiply
    #    by 10, then convert int.)
    # 2. There are currently 18 columns, and we can cut that way down. Most of the time-based
    #    columns should go. They specifically created an hour column, so maybe that is the
    #    only useful one.  All of the "static" columns should probably stay since they specifically
    #    created them.  Tracing through the preprocessing code in data_utils.py, can give us a
    #    sense of what they thought was useful.
    def tokenize(self, dataset: Tuple[pd.DataFrame, pd.DataFrame]) -> Tuple[set, set]:
        trainset, testset = dataset
        trainset_tokens = self.concat_dataframe_cols(trainset)
        testset_tokens = self.concat_dataframe_cols(testset)
        return trainset_tokens, testset_tokens
        