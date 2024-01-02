class GenericTokenizer:
    def __init__(self, **kwargs):
        special_tokens_dict = {
            'pad_token': '[PAD]',
            'bos_token': '[BOS]',
            'eos_token': '[EOS]',
        }
        token_to_id_map = {}
        id_to_token_map = {}


    def normalize(self, dataset):
        raise NotImplementedError()

    def preprocess(self, dataset):
        raise NotImplementedError()

    def model(self, dataset):
        raise NotImplementedError()

    def post_process(self, dataset):
        raise NotImplementedError()

    def encode(self, dataset):
        raise NotImplementedError()

    def decode(self, dataset):
        raise NotImplementedError()

