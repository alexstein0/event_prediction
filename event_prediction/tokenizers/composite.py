from .generic_tokenizer import GenericTokenizer
import pandas as pd
from typing import Tuple, Set

from event_prediction import data_utils

class Composite(GenericTokenizer):
    def __init__(self, tokenizer_cfgs, data_cfgs):
        super().__init__(tokenizer_cfgs, data_cfgs)

    def pretokenize(self, dataset):
        all_tokens = data_utils.concat_dataframe_cols(dataset[self.data_processor.all_cols])
        labels = dataset[self.data_processor.index_columns]
        return all_tokens, labels

    def model(self, dataset):
        # todo check if this is right way to do composite? words are the concat of the whole sentence
        # this is effectively a "word level" tokenizer.  most of the work is done by the pretokenizer and this simply maps inputs to IDS
        self.define_tokenization(dataset)
        return dataset

    def post_process(self, dataset, labels=None):
        special_tokens_added = []
        if labels is not None:
            index_tokens = []
            dataset.reset_index(drop=True, inplace=True)
            labels.reset_index(drop=True, inplace=True)
            for token in labels.columns:
                tok_locs = labels[token][labels[token] != labels[token].shift()].copy().astype(str)
                tok_locs[:] = token
                index_tokens.append(tok_locs)
                special_tokens_added.append(token)
            dataset = pd.concat([*index_tokens, dataset], axis=0).sort_index().reset_index(drop=True)

        for st in special_tokens_added:
            self.special_tokens_dict[st] = st
            self.add_token(st)

        return dataset

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
    def tokenize(self, dataset: Tuple[pd.DataFrame, pd.DataFrame]) -> Tuple[Set, Set]:
        trainset, testset = dataset
        trainset_tokens = data_utils.concat_dataframe_cols(trainset)
        testset_tokens = data_utils.concat_dataframe_cols(testset)
        return trainset_tokens, testset_tokens
