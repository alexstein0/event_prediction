from tokenizers import Tokenizer

class GenericTokenizer():
    def __init__(self):
        # super().__init__()
        pass

    def tokenize(self, dataset):
        raise NotImplementedError()
