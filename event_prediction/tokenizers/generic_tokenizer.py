import logging
from event_prediction import data_utils, get_data_processor
import torch
from typing import Set, List, Dict

log = logging.getLogger(__name__)

class GenericTokenizer:
    def __init__(self, tokenizer_cfgs, data_cfgs):
        self.special_tokens_dict = {
            'pad_token': '[PAD]',
            'bos_token': '[BOS]',
            'eos_token': '[EOS]',
            'unk_token': '[UNK]',
        }
        self.bos_token_id = None
        self.eos_token_id = None
        self.token_to_id = {}
        self.id_to_token = {}
        self.vocab = set()
        self.data_processor = get_data_processor(data_cfgs)

    def normalize(self, dataset):
        """Normalizes all the data in the table
        This includes:
            1. bucketing numeric values if doing that (maybe preprocessing?)
            2. adding any new values to the table (such as converting dollars, adding total minutes, etc)
        """
        log.info("Normalizing Data...")
        dataset = self.data_processor.normalize_data(dataset)
        return dataset

    def pretokenize(self, dataset):
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
            2. Word Piece
            3. Word Level
            4. Unigram
            """
        raise NotImplementedError()

    def post_process(self, dataset):
        return dataset
        # raise NotImplementedError()

    def encode(self, data: List[str]) -> torch.Tensor:
        if len(self.vocab) == 0:
            raise ValueError("Must create token ids first")
        output = torch.zeros(len(data))
        for i in range(len(data)):
            val = self.token_to_id.get(data[i], -1)  # todo (unknown tokens?)
            output[i] = val
        return output

    def decode(self, data: torch.Tensor) -> List[str]:
        if len(self.vocab) == 0:
            raise ValueError("Must create token ids first")
        output = []
        for i in data:
            val = self.id_to_token.get(i, self.special_tokens_dict['unk_token'])
            output.append(val)
        return output

    def define_tokenization(self, dataset: Set[str]):
        i = 0
        for key, val in self.special_tokens_dict.items():
            self.vocab.add(val)
            self.id_to_token[i] = val
            self.token_to_id[val] = i
            i += 1

        self.vocab.update(dataset)
        for val in dataset:
            self.id_to_token[i] = val
            self.token_to_id[val] = i
            i += 1

        self.update_special_token_ids()


    def update_special_token_ids(self):
        self.bos_token_id = self.token_to_id[self.special_tokens_dict['bos_token']]
        self.eos_token_id = self.token_to_id[self.special_tokens_dict['eos_token']]


    def save(self, file_name: str, tokenizer_dir: str):
        output = {}
        output["vocab"] = list(self.vocab)
        output["id_to_token"] = self.id_to_token
        output["token_to_id"] = self.token_to_id

        path = data_utils.save_json(output, tokenizer_dir, f"{file_name}.json")
        log.info(f"Saved tokenizer to {path}")


    def load(self, data: Dict):
        self.vocab = set(data["vocab"])
        self.id_to_token = data["id_to_token"]
        self.token_to_id = data["token_to_id"]
        self.update_special_token_ids()


    def load_vocab_from_file(self, file_name: str, tokenizer_dir: str):
        data = data_utils.read_json(tokenizer_dir, f"{file_name}.json")
        self.load(data)

