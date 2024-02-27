import hydra
import event_prediction
from event_prediction import model_utils, trainer_utils, data_utils, tokenizer_utils, get_data_processor
import logging
from typing import Dict
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer
import os
from hydra.utils import get_original_cwd
import datasets

import numpy as np
import multiprocessing

log = logging.getLogger(__name__)


def main_pretrain(cfg, setup=None) -> Dict:
    # tokenized_name = f"{cfg.data.name}_{cfg.tokenizer.name}"
    tokenized_name = cfg.tokenizer_name
    path = os.path.join(get_original_cwd(), cfg.tokenizer_dir, tokenized_name)
    log.info(f"Loading tokenizer from: {path}")
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)

    if cfg.preprocess_data_name is None:
        dataset = data_utils.get_data_from_raw(cfg.data, cfg.data_dir, False, False)
        data_processor = get_data_processor(cfg.data)
        dataset = data_utils.preprocess_dataset(dataset, data_processor, cfg.tokenizer.numeric_bucket_amount)

    else:
        data_dir = os.path.join(get_original_cwd(), cfg.data_dir)
        filepath = os.path.join(data_dir, cfg.preprocess_data_name)
        dataset = datasets.load_from_disk(filepath)

    # dataset = dataset.map(lambda example: tokenizer(example["text"]), batched=True)
    model = model_utils.get_model(
        cfg.model, tokenizer
    )
    model_size = sum(t.numel() for t in model.parameters())
    vocab_size = len(tokenizer.vocab)
    log.info(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")
    log.info(f"Vocab size: {vocab_size}")

    # tokenized_string_dataset = []
    # for x in dataset['text']:
    #     tokenized_string_dataset.extend(x.split())
    # # # todo doesnt work for atomic because it doesnt tokenize separately?
    # # # tokenized_string_dataset_labels = dataset['label']
    # # assert isinstance(tokenized_string_dataset, list), f"Expected list of string tokens, instead got {type(tokenized_string_dataset)}"
    # log.info(f"Total tokens in dataset: {len(tokenized_string_dataset)}")
    # log.info(f"Unique tokens in dataset: {len(set(tokenized_string_dataset))}")
    # log.info(f"Num tokens not in vocab: {len(set(tokenized_string_dataset)) - len(tokenizer.vocab)}")


    # HUGGINGFACE
    tokenizer.add_special_tokens({'pad_token': '[PAD]', 'unk_token': '[UNK]'})

    tokenized_string_dataset = data_utils.preprocess_and_tokenize_data(dataset, tokenizer)
    log.info("DATASET TOKENIZED")

    # Get info on the dataset need for classification and make a validation split that doesn't interrupt rows of of the original tabular data
    classification_info = tokenizer_utils.get_classification_options(tokenizer, label_in_last_col=True)
    num_cols = classification_info["num_cols"]
    label_ids = classification_info["label_ids"]
    train_ratio = cfg.model.train_ratio
    num_tokens = len(tokenized_string_dataset)
    split_point = int(num_tokens * train_ratio) - int(num_tokens * train_ratio) % num_cols 

    assert isinstance(tokenized_string_dataset,
                      list), f"Expected list of string tokens, instead got {type(tokenized_string_dataset)}"
    log.info(f"Total tokens in dataset: {num_tokens}")
    log.info(f"Unique tokens in dataset: {len(set(tokenized_string_dataset))}")
    log.info(f"Num tokens not in vocab: {len(set(tokenized_string_dataset)) - len(tokenizer.vocab)}")
    
    train_loader, val_loader = data_utils.get_dataloader(cfg.model, tokenizer, tokenized_string_dataset, split_point=split_point)
    log.info(f"Num train loader batches: {len(train_loader)}")
    log.info(f"Num val loader batches: {len(val_loader)}")
    log.info(f"Dataloader batch size: {train_loader.batch_size}")
    log.info(f"Model context length: {cfg.model.context_length}")
    log.info(
        f"Total tokens in dataloaders (n_batches * batch_sz * context_len): {(len(train_loader) + len(val_loader)) * val_loader.batch_size * cfg.model.context_length}")

    trainer = trainer_utils.get_trainer(cfg.model, model, train_loader, val_loader, num_cols, label_ids)

    log.info("TRAINING")
    model_path = trainer.train()
    log.info(f"Saving to {model_path}")

    metrics = {}
    return metrics



@hydra.main(
    config_path="event_prediction/config",
    config_name="pretrain",
    version_base="1.3",
)
def launch(cfg):
    event_prediction.utils.main_launcher(cfg, main_pretrain, job_name="pretrain")


if __name__ == "__main__":
    launch()
