import logging

import hydra

import event_prediction

log = logging.getLogger(__name__)


def main_eval(cfg, setup=None):
    print("HELLO EVAL")
    # Load model from checkpoint file specified in config
    model = event_prediction.model_utils.get_model(
        cfg.model, cfg.model_dir, cfg.tokenizer, cfg.data
    )
    # Load test data
    tokenizer, dataset = event_prediction.load_tokenizer_and_data(
        cfg.tokenizer_dir, cfg.data_dir, cfg.tokenizer, cfg.data
    )
    tokens = tokenizer.tokenize(dataset) 
    # Evaluate model
    output = model(tokens)

    


@hydra.main(config_path="event_prediction/config", config_name="eval_config", version_base="1.3")
def launch(cfg):
    event_prediction.utils.main_launcher(cfg, main_eval, job_name="eval")


if __name__ == "__main__":
    launch()
