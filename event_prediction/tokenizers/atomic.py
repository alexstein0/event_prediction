from .generic_tokenizer import GenericTokenizer
from event_prediction import data_utils
from typing import List

class Atomic(GenericTokenizer):
    def __init__(self, tokenizer_cfgs, data_cfgs):
        super().__init__(tokenizer_cfgs, data_cfgs)

    def pretokenize(self, dataset):
        exit()
        indexes = dataset[self.data_processor.get_index_columns()]
        prepended_tokens, special_tokens_added = data_utils.get_prepended_tokens(indexes)
        # dataset = data_utils.interweave_series([prepended_tokens, dataset])
        if len(self.data_processor.get_label_columns()) > 0:
            labels = dataset[self.data_processor.get_label_columns()]
        else:
            labels = None

        for st in special_tokens_added:
            self.special_tokens_dict[st] = st

        normal_rows = dataset["spec"].isnull()
        special_rows = dataset["spec"].notnull()
        special_table = dataset[special_rows]["spec"].to_frame()

        row_token = "ROW"
        col_token = "COL"
        main_table = dataset[normal_rows][self.data_processor.get_data_cols()]
        dataset = data_utils.add_special_tabular_tokens(main_table, add_col_sep=col_token, add_row_sep=row_token)
        self.special_tokens_dict[row_token] = row_token
        self.special_tokens_dict[col_token] = col_token

        all_tokens = data_utils.cols_to_words(dataset, special_table)

        return all_tokens, indexes, labels

    def model(self, dataset):
        # todo check if this is right way to do composite? words are the concat of the whole sentence
        # this is effectively a "word level" tokenizer.  most of the work is done by the pretokenizer and this simply maps inputs to IDS
        exit()
        self.add_special_tokens()
        self.add_all_tokens(set(dataset.values.tolist()))
        return dataset

    def post_process(self, dataset) -> List[str]:
        dataset = dataset.astype(str)
        return dataset.to_list()

