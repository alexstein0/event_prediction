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

    log.info("Normalizing...")
    start_time = time.time()
    normed_data = tokenizer.normalize(data)
    log.info(f"{f'Time taken for normalize:':40s} {time.time() - start_time:5.3f} seconds")

    log.info("Pretokenizing...")
    start_time = time.time()
    # wordified_data, indexes, labels = tokenizer.pretokenize(normed_data)
    wordified_data = tokenizer.pretokenize(normed_data)
    log.info(f"{f'Time taken for pretokenize:':40s} {time.time() - start_time:5.3f} seconds")

    log.info("Applying tokenization model...")
    start_time = time.time()
    tokenized_data = tokenizer.model(wordified_data)
    log.info(f"{f'Time taken for model:':40s} {time.time() - start_time:5.3f} seconds")

    log.info("Postprocessing...")
    start_time = time.time()
    post_processed_data = tokenizer.post_process(tokenized_data)
    log.info(f"{f'Time taken for post_process:':40s} {time.time() - start_time:5.3f} seconds")

    log.info("Saving processed dataset...")
    start_time = time.time()
    dataset = data_utils.create_dataset(post_processed_data)
    data_utils.save_processed_dataset(dataset, cfg.processed_data_dir, cfg.data.name)
    log.info(f"{f'Time taken for save_processed_dataset:':40s} {time.time() - start_time:5.3f} seconds")

    tokenizer.training_complete()

    log.info("Saving tokenizer...")
    start_time = time.time()
    tokenizer.save(cfg.data.name, cfg.tokenizer_dir)
    log.info(f"{f'Time taken for save:':40s} {time.time() - start_time:5.3f} seconds")

    metrics = tokenizer.get_metrics()
    log.info("OUTPUT METRICS")
    for k, v in metrics.items():
        log.info(f"{k+':':20s} {str(v):10s}")

    return metrics

@hydra.main(config_path="event_prediction/config", config_name="pre_process_data", version_base="1.3")
def launch(cfg):
    event_prediction.utils.main_launcher(cfg, main_process_data, job_name="process-data")


if __name__ == "__main__":
    launch()

