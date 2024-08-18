import hydra
from datasets import Dataset

import event_prediction
from event_prediction import get_data_processor, data_preparation
import logging
from typing import Dict, List
import os
from hydra.utils import get_original_cwd


log = logging.getLogger(__name__)


def create_text_dataset(cfg, setup=None) -> Dict:
    log.info(f"STARTING TRAINING")

    # log.info(f"GET TOKENIZER")
    # tokenized_name = cfg.tokenizer_name
    # tokenizer_path = os.path.join(get_original_cwd(), cfg.tokenizer_dir, tokenized_name)
    # log.info(f"Loading tokenizer from: {tokenizer_path}")
    # tokenizer = event_prediction.load_tokenizer(tokenizer_path)
    # log.info(f"TOKENIZER LOAD COMPLETE")

    log.info(f"GET DATA")
    data_processor = get_data_processor(cfg.data)
    data = data_preparation.get_data_and_dont_tokenize(cfg, data_processor)
    log.info(f"DATASET AS TEXT")

    log.info(f"Sample: {data[0]}")
    log.info(f"Index cols: {data_processor.get_index_columns()}")
    log.info(f"Label cols: {data_processor.get_label_columns()}")
    log.info(f"Num cols: {len(data_processor.get_data_cols())} | {data_processor.get_data_cols()}")
    # log.info("Column keys: " + " ".join([f"{k}: {v}" for k, v in col_to_id_dict.items()]))

    tokenized_data = data_preparation.split_data_by_column(data, "User")
    if cfg.data.split is not None:
        tokenized_data = data_preparation.create_train_test_split(tokenized_data, cfg.model.train_test_split)
        tokenized_data = tokenized_data[cfg.data.split]
    log.info(f"Data has {sum([len(x) for user, x in tokenized_data.items()])} samples")
    log.info("SPLITTTED DATA BY USER AND TEST SPLIT")
    seq_length = cfg.model.seq_length
    # num_rows = sum([x for x in tokenized_data.num_rows.values()])
    user_ids = list(set(tokenized_data.keys()))  # needs to be split by user already

    def concatenate_text(user_ds) -> List[str]:
        # user_rows = tokenized_dataset[uid]["text"]
        sequences = []
        user_rows = user_ds["text"]
        num_rows = len(user_rows)
        for starting_row_id in range(0, num_rows, seq_length):  # consider not striding by seq_len
            seq = user_rows[starting_row_id: starting_row_id + seq_length]
            seq = " ".join(seq)
            sequences.append(seq)
        return sequences

    log.info("ITERATING THROUGH USERS")
    all_sequences = []
    for uid in user_ids:
        user_info = tokenized_data[uid]
        # rows = user_info.map(lambda examples: concatenate_text(examples), batched=True, batch_size=len(user_info))
        rows = concatenate_text(user_info)
        all_sequences.extend(rows)

    output_dataset = Dataset.from_dict({'text': all_sequences})

    # dataloaders = {"test": data_preparation.prepare_text_dataloader(tokenized_data, tokenizer, cfg)}
    data_dir = os.path.join(get_original_cwd(), cfg.processed_data_dir)
    save_name = cfg.data.name
    if cfg.data.save_name is not None:
        save_name = cfg.data.save_name
    filepath = os.path.join(data_dir, save_name)
    data_preparation.save_dataset(output_dataset, filepath)
    # tokenized_data.save_to_disk(filepath, num_proc=threads)
    log.info(f"DATASET SAVED to {filepath}")
    return {}


@hydra.main(
    config_path="event_prediction/config",
    config_name="pretrain",
    version_base="1.3",
)
def launch(cfg):
    event_prediction.utils.main_launcher(cfg, create_text_dataset, job_name="tokenize_dataset")


if __name__ == "__main__":
    launch()
