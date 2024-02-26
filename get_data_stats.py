import logging
import time
from typing import Dict
import datasets
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers, Regex, processors
from transformers import PreTrainedTokenizerFast
import os
import hydra

import event_prediction
from event_prediction import data_utils, get_data_processor
import multiprocessing

log = logging.getLogger(__name__)

def main_process_data(cfg, setup=None) -> Dict:
    dataset = data_utils.get_data_from_raw(cfg.data, cfg.data_dir, cfg.save_tar, cfg.save_csv)

    log.info("DATA INFO")
    log.info(f"Rows: {len(dataset)}")

    data_processor = get_data_processor(cfg.data)

    dataset = data_processor.normalize_data(dataset)
    for col in data_processor.get_numeric_columns():
        dataset[col], buckets = data_utils.convert_to_binary_string(dataset[col], cfg.tokenizer.numeric_bucket_amount)

    col_id = 0
    for col in data_processor.get_all_cols():
        # if col in data_processor.get_numeric_columns():
        #     continue
        # else:
        dataset[col] = str(col_id) + "_" + dataset[col].astype(str)
        col_id += 1

    dataset = datasets.Dataset.from_pandas(dataset)

    def concat_columns(example):
        new_ex = {}
        new_ex["text"] = " ".join(example.values())
        return new_ex

    dataset = dataset.map(lambda example: example, batched=True)
    try:
        threads = max(os.cpu_count(), multiprocessing.cpu_count(), 1)
    except:
        threads = 1
    dataset = dataset.map(concat_columns, num_proc=threads)
    dataset = dataset.select_columns("text")

    log.info(f"num cols: {len(data_processor.get_all_cols())} | {data_processor.get_all_cols()}")
    log.info(f"Vocab_size: {tokenizer.get_vocab_size()}")
    tok_name = f"{cfg.data.name}_{cfg.tokenizer.name}"
    wrapped_tokenizer.save_pretrained(os.path.join(cfg.tokenizer_dir, tok_name))
    return {}

@hydra.main(config_path="event_prediction/config", config_name="pre_process_data", version_base="1.3")
def launch(cfg):
    event_prediction.utils.main_launcher(cfg, main_process_data, job_name="get_data_stats")


if __name__ == "__main__":
    launch()

