import logging
import time
from typing import Dict

import hydra

import event_prediction
from event_prediction import data_utils

log = logging.getLogger(__name__)


def main_process_data(cfg, setup=None) -> Dict:
    data = data_utils.get_data_from_raw(cfg.data, cfg.data_dir, cfg.save_tar, cfg.save_csv)
    tokenizer = event_prediction.get_tokenizer(cfg.tokenizer, cfg.data)
    # normed_data = tokenizer.normalize(data)
    # wordified_data = tokenizer.pretokenize(normed_data)
    # tokenized_data = tokenizer.model(wordified_data)
    # post_processed_data = tokenizer.post_process(tokenized_data)
    # data_utils.save_processed_dataset(post_processed_data, cfg.data, cfg.processed_data_dir)
    # tokenizer.save(cfg.data.name, cfg.tokenizer_dir)

    log.info("Normalizing...")
    start_time = time.time()
    normed_data = tokenizer.normalize(data)
    log.info(f"Time taken for normalize: {time.time() - start_time} seconds")

    log.info("Pretokenizing...")
    start_time = time.time()
    wordified_data = tokenizer.pretokenize(normed_data)
    log.info(f"Time taken for pretokenize: {time.time() - start_time} seconds")

    log.info("Applying tokenization model...")
    start_time = time.time()
    tokenized_data = tokenizer.model(wordified_data)
    log.info(f"Time taken for model: {time.time() - start_time} seconds")

    log.info("Postprocessing...")
    start_time = time.time()
    post_processed_data = tokenizer.post_process(tokenized_data)
    log.info(f"Time taken for post_process: {time.time() - start_time} seconds")

    log.info("Saving processed dataset...")
    start_time = time.time()
    data_utils.save_processed_dataset(post_processed_data, cfg.data, cfg.processed_data_dir)
    log.info(f"Time taken for save_processed_dataset: {time.time() - start_time} seconds")

    log.info("Saving tokenizer...")
    start_time = time.time()
    tokenizer.save(cfg.data.name, cfg.tokenizer_dir)
    log.info(f"Time taken for save: {time.time() - start_time} seconds")

    # train_dataset, test_dataset = data_utils.get_train_test_split(dataset, cfg.data.train_test_split_year)
    return {}

@hydra.main(config_path="event_prediction/config", config_name="pre_process_data", version_base="1.3")
def launch(cfg):
    event_prediction.utils.main_launcher(cfg, main_process_data, job_name="process-data")


if __name__ == "__main__":
    launch()

