import hydra
import event_prediction
from event_prediction import utils, data_preparation, ModelTrainerInterface
import logging
from typing import Dict
import os
from hydra.utils import get_original_cwd
import time

log = logging.getLogger(__name__)


def main_pretrain(cfg, setup=None) -> Dict:
    log.info(f"------------- STARTING TRAINING -------------")
    initial_time = time.time()
    metrics = {}

    # GET TOKENIZER
    log.info(f"------------- GETTING TOKENIZER -------------")
    section_timer = time.time()
    tokenized_name = cfg.tokenizer_name
    tokenizer_path = os.path.join(get_original_cwd(), cfg.tokenizer_dir, tokenized_name)
    log.info(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = event_prediction.load_tokenizer(tokenizer_path)
    elapsed_times = utils.get_time_deltas(section_timer, initial_time)
    log.info(f"Total time: {elapsed_times[0]} ({elapsed_times[1]}  overall)")
    log.info(f"------------- TOKENIZER LOAD COMPLETE -------------")

    log.info(f"------------- GET DATA -------------")
    section_timer = time.time()
    data_dir = os.path.join(get_original_cwd(), cfg.processed_data_dir)
    if cfg.tokenized_data_name is None:
        log.info("No tokenized data name provided, exiting")
        return {}
    filepath = os.path.join(data_dir, cfg.tokenized_data_name)
    log.info(f"Loading data from: {filepath}")
    tokenized_data = event_prediction.get_data(filepath)
    elapsed_times = utils.get_time_deltas(section_timer, initial_time)
    log.info(f"Total time: {elapsed_times[0]} ({elapsed_times[1]}  overall)")
    log.info(f"------------- DATASET LOAD COMPLETE -------------")

    # PROCESS DATA
    log.info(f"------------- PROCESS TOKENIZED DATASET -------------")
    section_timer = time.time()
    tokenized_data = data_preparation.split_data_by_column(tokenized_data, "User")
    tokenized_data = data_preparation.create_train_test_split(tokenized_data, cfg.model.train_test_split)
    dataloaders = data_preparation.prepare_dataloaders(tokenized_data, tokenizer, cfg)

    train_loader = dataloaders["train"]
    val_loader = dataloaders["test"]
    # train_users = train_loader.dataset.user_ids
    # val_users = val_loader.dataset.user_ids
    # metrics["train_users"] = train_users
    # metrics["val_users"] = val_users

    log.info(f"Num train loader batches: {len(train_loader)}")
    log.info(f"Num val loader batches: {len(val_loader)}")
    log.info(f"Number of Columns: {train_loader.dataset.num_columns}")
    sample_size = train_loader.dataset.data.shape[1]
    log.info(f"Sequence length (rows): {cfg.model.seq_length}")
    log.info(f"Elements per sample (columns * sequence length): {sample_size}")
    log.info(f"Batch Size: {train_loader.batch_size}")
    log.info(f"Total tokens in dataloaders (n_batches * batch_sz * seq_length * columns): "
             f"{(len(train_loader) + len(val_loader)) * val_loader.batch_size * cfg.model.seq_length * sample_size:,}")
    elapsed_times = utils.get_time_deltas(section_timer, initial_time)
    log.info(f"Total time: {elapsed_times[0]} ({elapsed_times[1]}  overall)")
    log.info(f"------------- DATASET PROCESSED -------------")

    # log.info("NEED MORE DATASET PROCESSING, split into batches/mini batches etc")

    # todo
    # if mlm add mask token and pad token
    # tokenizer.add_special_tokens(
    #     {'pad_token': '[PAD]',
    #      'unk_token': '[UNK]',
    #      'new_row': '[ROW]'}
    # )

    # label_col_id = "15"
    # train model
    # GET MODEL
    log.info(f"------------- INITIALIZING MODEL -------------")
    section_timer = time.time()
    classification = True
    if classification:
        # todo add collator
        # classification_info = tokenizer_utils.get_classification_options(tokenizer, label_in_last_col=True) #, label_col_prefix=label_col_id)
        # num_cols = classification_info["num_cols"]
        # label_ids = classification_info["label_ids"]
        model_interface = ModelTrainerInterface(cfg, tokenizer, dataloaders)

    else:
        # todo this doesnt work
        data_collator = datacollator.TransDataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=cfg.mask_prob
        )
        model_interface = ModelTrainerInterface(cfg.model, tokenizer, dataloaders, num_cols=num_cols, label_ids=label_ids)

    model = model_interface.model
    model_size = sum(t.numel() for t in model.parameters())
    vocab_size = len(tokenizer.vocab)
    log.info(f"Model Name: {model.model.name_or_path}")
    log.info(f"Model size: {model_size/1000**2:.1f}M parameters")
    log.info(f"Vocab size: {vocab_size}")
    elapsed_times = utils.get_time_deltas(section_timer, initial_time)
    log.info(f"Total time: {elapsed_times[0]} ({elapsed_times[1]}  overall)")
    log.info(f"------------- LOADING COMPLETE -------------")
    log.info(f"------------- BEGIN TRAINING -------------")
    section_timer = time.time()
    eval_metrics, train_metrics = model_interface.train()
    metrics.update(eval_metrics)
    # train_metrics = model_interface.train_HF(dataloaders["train"])
    model_path = model_interface.save_model("final")
    log.info(f"Saving to {model_path}")
    elapsed_times = utils.get_time_deltas(section_timer, initial_time)
    log.info(f"Total time: {elapsed_times[0]} ({elapsed_times[1]}  overall)")
    log.info(f"------------- TRAINING COMPLETE -------------")

    return metrics


@hydra.main(
    config_path="event_prediction/config",
    config_name="pretrain",
    version_base="1.3",
)
def launch(cfg):
    utils.main_launcher(cfg, main_pretrain, job_name="pretrain")


if __name__ == "__main__":
    launch()