import hydra
import event_prediction
from event_prediction import data_utils
import logging
from typing import Dict

log = logging.getLogger(__name__)

def main_process_data(cfg, setup=None) -> Dict:
    data = data_utils.get_data_from_raw(cfg.data, cfg.data_dir)
    log.info("Doing basic preprocessing...")
    dataset = data_utils.prepare_dataset(cfg.data, data)
    print(dataset.columns)

    # TODO
    tokenizer = event_prediction.get_tokenizer(cfg.tokenizer)  # todo pretrain tokenize
    # normalize
    # pre-process
    tokenizer.create_elementary_tokens(dataset, cfg.data)
    # model
    # post-process
    # save

    log.info("Saving File")
    filepath = data_utils.save_processed_dataset(dataset, cfg.data, cfg.processed_data_dir)
    log.info(f"Dataset saved to {filepath}")
    log.info("Splitting into train and test sets...")
    train_dataset, test_dataset = data_utils.get_train_test_split(dataset, cfg.data.train_test_split_year)
    return {}

@hydra.main(config_path="event_prediction/config", config_name="pre_process_data", version_base="1.3")
def launch(cfg):
    event_prediction.utils.main_launcher(cfg, main_process_data, job_name="process-data")


if __name__ == "__main__":
    launch()