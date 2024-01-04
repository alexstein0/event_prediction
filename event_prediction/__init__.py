"""Initialize event_predictions"""

from event_prediction import utils
from event_prediction.data import data_utils, get_data_processor
from event_prediction.models import model_utils, trainer_utils
from event_prediction.tokenizers import get_tokenizer, GenericTokenizer

__all__ = [
    "utils",
    "data_utils",
    "model_utils",
    "trainer_utils",
    "get_data_processor",
    "get_tokenizer"
]

def get_tokenizer_and_data(tokenizer_dir, data_dir, tokenizer_cfg, data_cfg) -> (GenericTokenizer, str):

    tokenizer = get_tokenizer(tokenizer_cfg, data_cfg)
    tokenizer_data = data_utils.read_json(tokenizer_dir, data_cfg.name)
    tokenizer.load(tokenizer_data)
    dataset = data_utils.load_dataset(data_cfg.name, data_dir)
    return tokenizer, dataset