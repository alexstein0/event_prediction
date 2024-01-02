import logging
from event_prediction import data_utils, get_data_processor

log = logging.getLogger(__name__)

class GenericTokenizer:
    def __init__(self, tokenizer_cfgs, data_cfgs):
        special_tokens_dict = {
            'pad_token': '[PAD]',
            'bos_token': '[BOS]',
            'eos_token': '[EOS]',
        }
        token_to_id_map = {}
        id_to_token_map = {}
        self.data_processor = get_data_processor(data_cfgs)

    def normalize(self, dataset):
        """Normalizes all the data in the table
        This includes:
            1. bucketing numeric values if doing that (maybe thats preprocessing?)
            2. adding any new values to the table (such as converting dollars, adding total minutes, etc)
        """
        log.info("Normalizing Data...")
        dataset = self.data_processor.normalize_data(dataset)
        return dataset

    def preprocess(self, dataset):
        """This is where the different tokenizers will convert the tables into 'sentences' of 'words'
        The words outputted from here can be:
            1. Composite tokens with goal of predicting next composite token
            2. atomic tokens with the goal of predicting the next set of atomic tokens
            3. The tokens can actually be embedding vectors
        so the goal here is to create those sentences to be passed.  The output here should be agnostic to the problem (of tabular data)"""
        # log.info("Preprocessing data")
        # dataset = self.data_processor.preprocess_data(dataset)
        # return dataset
        raise NotImplementedError()


    def model(self, dataset):
        """Tokenization here consists of taking the previous 'sentences' and doing actual tokenization such as:
            1. BPE
            2. Word Piece"""
        raise NotImplementedError()

    def post_process(self, dataset):
        raise NotImplementedError()

    def encode(self, dataset):
        raise NotImplementedError()

    def decode(self, dataset):
        raise NotImplementedError()

