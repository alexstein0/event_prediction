import hydra
import event_prediction
import logging
from typing import Dict

log = logging.getLogger(__name__)


def main_pretrain(cfg, setup=None) -> Dict:
    print("HELLO WORLD")
    data = event_prediction.data_utils.get_data(cfg.data, cfg.data_dir)
    dataset = event_prediction.data_utils.prepare_dataset(cfg.data, data)
    tokenizer = event_prediction.get_tokenizer(cfg.tokenizer)
    tokens = tokenizer.tokenize(dataset)
    # model = # TODO

    return {}


@hydra.main(config_path="event_prediction/config", config_name="pre_train_model_config", version_base="1.3")
def launch(cfg):
    event_prediction.utils.main_launcher(cfg, main_pretrain, job_name="pre-train")


if __name__ == "__main__":
    launch()
