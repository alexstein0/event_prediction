import logging

import hydra
from hydra.utils import get_original_cwd
import torch
import event_prediction
from event_prediction import data_utils, tokenizer_utils, trainer_utils
import os

log = logging.getLogger(__name__)


def main_eval(cfg, setup=None):
    log.info(f"STARTING EVAL")

    # GET TOKENIZER
    log.info(f"GET TOKENIZER")
    tokenized_name = cfg.tokenizer_name
    tokenizer_path = os.path.join(get_original_cwd(), cfg.tokenizer_dir, tokenized_name)
    log.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = event_prediction.load_tokenizer(tokenizer_path)
    log.info(f"TOKENIZER LOAD COMPLETE")

    # GET DATA
    log.info(f"GET DATA")
    if cfg.tokenized_data_name is None:
        data_processor = event_prediction.get_data_processor(cfg.data)
        tokenized_data = data_utils.get_data_and_tokenize(cfg, data_processor, tokenizer)

        if cfg.save_csv or cfg.preprocess_only:
            data_dir = os.path.join(get_original_cwd(), cfg.processed_data_dir)
            filepath = os.path.join(data_dir, cfg.data.name)
            tokenized_data.save_to_disk(filepath)
            log.info(f"DATASET SAVED to {filepath}")
            if cfg.preprocess_only:
                return {}
    else:
        data_dir = os.path.join(get_original_cwd(), cfg.processed_data_dir)
        filepath = os.path.join(data_dir, cfg.tokenized_data_name)
        tokenized_data = event_prediction.get_data(filepath)
        log.info(f"DATASET LOADED from {filepath}")

    log.info(f"DATASET LOAD COMPLETE")

    # PROCESS DATA
    log.info(f"PROCESS DATASET")
    tokenized_string_dataset = tokenized_data["input_ids"]
    train_loader, val_loader = data_utils.prepare_dataloaders(cfg.model, tokenizer, tokenized_string_dataset)
    dataloaders = {"test": val_loader}
    log.info(f"Num test loader batches: {len(val_loader)}")
    log.info(f"Dataloader batch size: {val_loader.batch_size}")
    log.info(f"Model context length: {cfg.model.context_length}")
    log.info(f"Total tokens in dataloaders (n_batches * batch_sz * context_len): "
             f"{len(val_loader) * val_loader.batch_size * cfg.model.context_length}")
    log.info(f"DATASET PROCESSED")

    # Evaluate model
    # GET MODEL
    log.info(f"INITIALIZING MODEL")
    classification = True
    if classification:
        # todo add collator
        classification_info = tokenizer_utils.get_classification_options(tokenizer, label_in_last_col=True)
        num_cols = classification_info["num_cols"]
        label_ids = classification_info["label_ids"]
        model_interface = trainer_utils.get_trainer(cfg.model, tokenizer, dataloaders, num_cols=num_cols, label_ids=label_ids)

    else:
        raise NotImplementedError()

    model = model_interface.get_model()
    model_size = sum(t.numel() for t in model.parameters())
    vocab_size = len(tokenizer.vocab)
    log.info(f"Model Name: {model.model.name_or_path}")
    log.info(f"Model size: {model_size / 1000 ** 2:.1f}M parameters")
    log.info(f"Vocab size: {vocab_size}")

    log.info(f"Loading complete.  Running Eval")
    losses = model_interface.test()

    return {}


@hydra.main(config_path="event_prediction/config", config_name="eval", version_base="1.3")
def launch(cfg):
    event_prediction.utils.main_launcher(cfg, main_eval, job_name="eval")


if __name__ == "__main__":
    launch()
