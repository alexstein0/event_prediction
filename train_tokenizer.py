import logging
import time
from typing import Dict
import datasets
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers, Regex, processors
from transformers import PreTrainedTokenizerFast
import os
import hydra
from hydra.utils import get_original_cwd

import event_prediction
from event_prediction import data_utils, get_data_processor, data_preparation
import multiprocessing

log = logging.getLogger(__name__)


def main_process_data(cfg, setup=None) -> Dict:
    log.info(f"Retrieving dataset from: {cfg.data}")
    dataset = data_utils.get_data_from_raw(cfg.data, cfg.data_dir)
    data_processor = get_data_processor(cfg.data)
    log.info(f"DATASET LOADED")
    dataset, col_to_id_dict = data_preparation.preprocess_dataset(dataset, data_processor, cfg.tokenizer.numeric_bucket_amount)
    log.info(f"DATASET processed")
    dataset = data_preparation.convert_to_huggingface(dataset, data_processor)
    log.info(f"DATASET converted to huggingface")

    unk_token = "[UNK]"
    tokenizer = Tokenizer(models.WordLevel(unk_token=unk_token))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

    trainer = trainers.WordLevelTrainer(
        vocab_size=cfg.tokenizer.vocab_size,
        special_tokens=['[PAD]',
                        unk_token,
                        # '[ROW]'
                        ]
        # special_tokens=list(set(special_token_args.values()))
    )

    def data_generator(batch_size=1024):

        len_dataset = len(dataset)
        for i in range(0, len_dataset, batch_size):
            rows = dataset[i: i + batch_size]
            yield rows['text']

    tokenizer.train_from_iterator(data_generator(), trainer=trainer, length=len(dataset))

    # # todo does this work?
    # processed_data_dir = os.path.join(cfg.processed_data_dir, cfg.data.name)
    # tokenized_data.save_to_disk(processed_data_dir)

    # Wrap into fast codebase
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        # model_max_length=cfg.data.seq_length,
        # **special_token_args,
    )

    wrapped_tokenizer.add_special_tokens(
        {'pad_token': '[PAD]',
         'unk_token': '[UNK]',
         # 'eos_token': '[ROW]'
         }
    )

    log.info("TRAINING COMPLETE")
    log.info(f"Vocab_size: {tokenizer.get_vocab_size()}")
    tok_name = f"{cfg.data.name}_{cfg.tokenizer.name}"
    files1 = wrapped_tokenizer.save_pretrained(tok_name)
    log.info(f"Saved tokenizer to {os.getcwd()}/{tok_name}")
    if cfg.tokenizer_dir is not None:
        tokenizer_path = os.path.join(get_original_cwd(), cfg.tokenizer_dir, tok_name)
        files2 = wrapped_tokenizer.save_pretrained(tokenizer_path)
        log.info(f"Also saved tokenizer to {tokenizer_path}")

    return {}


@hydra.main(config_path="event_prediction/config", config_name="pre_process_data", version_base="1.3")
def launch(cfg):
    event_prediction.utils.main_launcher(cfg, main_process_data, job_name="process-data")


if __name__ == "__main__":
    launch()

