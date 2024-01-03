import hydra
import event_prediction
import logging
from typing import Dict

log = logging.getLogger(__name__)


def main_pretrain(cfg, setup=None) -> Dict:
    print("HELLO WORLD")
    # dataset = event_prediction.data_utils.load_dataset(cfg.data, cfg.data_dir)
    tokenizer, dataset = event_prediction.get_tokenizer_and_data(
        cfg.tokenizer_dir, cfg.data_dir, cfg.tokenizer, cfg.data
    )  # todo pretrain tokenize
    tokens = tokenizer.tokenize(dataset)
    model = event_prediction.model_utils.get_model(
        cfg.model, tokenizer, cfg.data.context_length
    )
    trainer = event_prediction.trainer_utils.get_trainer(
        model=model,
        tokenizer=tokenizer,
        tokens=tokens,
        args=cfg.trainer,
    )
    trainer.train()
    return model


@hydra.main(
    config_path="event_prediction/config",
    config_name="process_data_config",
    version_base="1.3",
)
def launch(cfg):
    event_prediction.utils.main_launcher(cfg, main_pretrain, job_name="pre-train")


if __name__ == "__main__":
    launch()
