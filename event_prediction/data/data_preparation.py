import pandas as pd
from transformers import DataCollatorForLanguageModeling
from typing import List, Union, Dict, Tuple
import torch
from transformers.data.data_collator import _torch_collate_batch
import os
from datasets.distributed import split_dataset_by_node
import datasets
from datasets import Dataset, DatasetDict
from omegaconf import DictConfig
from torch.utils import data
from transformers import AutoTokenizer
from event_prediction import utils
import logging

from .data_utils import get_data_from_raw, convert_to_binary_string
import random
import json

log = logging.getLogger(__name__)
threads = utils.get_cpus()


# THIS CODE IS COPIED FROM FATA-TRANS  This is the base class for masked collator when no temporal component and no static/dynamic split
class TransDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def __call__(self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_ids = [x["input_ids"] for x in examples]
        # labels = [x["targets"] for x in examples]
        mask = [x["mask"] for x in examples]
        input_ids = _torch_collate_batch(input_ids, self.tokenizer)
        # labels = _torch_collate_batch(labels, self.tokenizer)
        mask = _torch_collate_batch(mask, self.tokenizer)
        sz = input_ids.shape
        if self.mlm:
            batch = input_ids.view(sz[0], -1)
            inputs, labels = self.mask_tokens(batch)
            # print("MLM label shape: ", labels.view(sz).shape)
            # print("MLM batch shape: ", inputs.view(sz).shape)
            return {"input_ids": inputs.view(sz), "masked_lm_labels": labels.view(sz)}
        else:
            # labels = labels.clone().detach()
            # if self.tokenizer.pad_token_id is not None:
            #     labels[labels == self.tokenizer.pad_token_id] = -100
            # return {"input_ids": input_ids, "targets": labels, "mask": mask}
            return {"input_ids": input_ids, "mask": mask}

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove "
                "the --mlm flag if you want to use this tokenizer. "
            )
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability
        # defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


# this code is adapted from cramming
class FastDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def __init__(self, *args, create_labels_entry=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.mlm = False
        self.create_labels_entry = create_labels_entry

    def torch_call(self, examples):
        """Simplified call assuming all dicts in the list of examples have the same layout and contain tensors.
        Assume further that all these tensors contain vectors of Long Tensors  [AND THEY HAVE TO BE LONG]"""
        if isinstance(examples[0], torch.Tensor):
            examples = [{"input_ids": ex} for ex in examples]
        # So this is the handmade version
        batch = dict()
        for key in examples[0].keys():
            elem = torch.as_tensor(examples[0][key])
            # block = examples[0][key].new_empty(len(examples), *examples[0][key].shape)
            # for idx, example in enumerate(examples):
            #     block[idx] = example[key]
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # storage = elem._storage()._new_shared(len(examples) * 8 * elem.shape[0], device=elem.device)  # 8 for byte->long
                # storage = elem.untyped_storage()._new_shared(len(examples) * 8 * elem.shape[0], device=elem.device)  # 8 for byte->long
                # out = elem.new(storage).resize_(len(examples), elem.shape[0])
                storage = elem._typed_storage()._new_shared(len(examples) * elem.shape[0], device=elem.device)
                out = elem.new(storage).resize_(len(examples), elem.shape[0])

            batch[key] = torch.stack([torch.as_tensor(example[key]) for example in examples], 0, out=out).contiguous()

        if self.create_labels_entry:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch


def prepare_dataloaders(tokenized_dataset: DatasetDict | Dataset, tokenizer, cfg: DictConfig) -> Dict[str, data.DataLoader]:
    """
    Takes in pretokenized hf dataset
    """

    # train_loader = data.DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True)
    # val_loader = data.DataLoader(val_data, batch_size=cfg.batch_size)
    if isinstance(tokenized_dataset, Dataset):
        test_split = cfg.model.train_test_split
        split_by_column = "User"
        tokenized_dataset = create_train_test_split_by_column(tokenized_dataset, test_split, split_by_column)
    train_data = tokenized_dataset["train"]
    val_data = tokenized_dataset["test"]
    train_loader = prepare_pretraining_dataloader(train_data, tokenizer, cfg)
    val_loader = prepare_validation_dataloader(val_data, tokenizer, cfg)
    return {"train": train_loader, "test": val_loader}


def get_dataset_and_collator(tokenized_dataset: Dataset, tokenizer, cfg: DictConfig) -> Tuple:
    if cfg.model.training_objective == "causal":
        tokenized_dataset = NextTokenPredictionDataset(tokenized_dataset, cfg.model.seq_length, tokenizer.pad_token_id, cfg.model.randomize_order, cfg.model.fixed_cols)
        collate_fn = TransDataCollatorForLanguageModeling(tokenizer=tokenizer, pad_to_multiple_of=cfg.impl.pad_to_multiple_of, mlm=False)

    elif cfg.model.training_objective == "masked":
        tokenized_dataset = MaskedLanguageModelingDataset(tokenized_dataset, n_cols)
        collate_fn = FastDataCollatorForLanguageModeling(tokenizer=tokenizer, pad_to_multiple_of=cfg.impl.pad_to_multiple_of, mlm=True)
    else:
        raise ValueError(f"training_objective must be 'causal' or 'masked', not {cfg.training_objective}")
    return tokenized_dataset, collate_fn


def prepare_pretraining_dataloader(tokenized_dataset: Dataset, tokenizer, cfg) -> data.DataLoader:
    tokenized_dataset, collate_fn = get_dataset_and_collator(tokenized_dataset, tokenizer, cfg)
    loader = to_train_dataloader(tokenized_dataset, collate_fn, cfg.model.batch_size)
    return loader


def prepare_validation_dataloader(tokenized_dataset: Dataset, tokenizer, cfg) -> data.DataLoader:
    tokenized_dataset, collate_fn = get_dataset_and_collator(tokenized_dataset, tokenizer, cfg)
    loader = to_val_dataloader(tokenized_dataset, collate_fn, cfg.model.batch_size)
    return loader


def preprocess_dataset(dataset: pd.DataFrame, data_processor, numeric_bucket_amount: int = 5) -> Tuple[pd.DataFrame, Dict[str, int]]:
    dataset = data_processor.normalize_data(dataset)
    data_processor.summarize_dataset(dataset, numeric_bucket_amount)
    for col in data_processor.get_numeric_columns():
        strings, buckets = convert_to_binary_string(dataset[col], numeric_bucket_amount)
        if len(strings[0]) > 0:
            dataset[col] = strings  # maybe a cleaner way to do this

    # todo right here is where we would extract the labels as well.
    #  Here we prepare string to be tokenized and below we will create string to tokenize
    all_cols = data_processor.get_data_cols()
    # all_cols.append(row_key)
    # todo right now the order of the columns must be the same at tokenization train/test
    dataset, col_to_id_dict = get_col_to_id_dict(all_cols, dataset)
    return dataset, col_to_id_dict


def get_col_to_id_dict(all_cols: List[str], dataset: pd.DataFrame = None):
    col_id = 0
    col_to_id_dict = {}
    for col in all_cols:
        col_to_id_dict[col] = col_id
        if dataset is not None:
            dataset[col] = str(col_id) + "_" + dataset[col].astype(str)
        col_id += 1

    return dataset, col_to_id_dict


def convert_to_huggingface(dataset: pd.DataFrame, data_processor) -> datasets.Dataset:
    row_token = "[ROW]"
    columns = dataset.columns
    dataset = Dataset.from_pandas(dataset, preserve_index=False)

    # dataset = dataset.map(lambda example: example, batched=True)
    def concat_columns(example):
        new_ex = {}
        # print([example[x] for x in example.keys() if x not in data_processor.get_index_columns()])
        row = " ".join([example[x] for x in example.keys() if x not in data_processor.get_index_columns()])
        # todo add this to post process instead?
        row = row + f" {row_token}"
        new_ex["text"] = row
        return new_ex

    dataset = dataset.map(concat_columns, num_proc=threads)
    dataset = dataset.select_columns(["text", *data_processor.get_index_columns()])
    return dataset


def get_start_end_indices(ds: Dataset) -> Dict[str, List[int]]:
    ranges = {}
    current_user_id = None
    start = 0
    cur = 0

    for row in ds:
        if row["User"] != current_user_id:
            if cur > 0:
                ranges[current_user_id] = [start, cur]
            current_user_id = row["User"]
            start = cur
        cur += 1

    ranges[current_user_id] = [start, cur]
    return ranges


def split_data_by_column(dataset: Dataset, split_col: str):
    start_end_dict = get_start_end_indices(dataset.select_columns([split_col]))
    output_dict = {}
    for k, v in start_end_dict.items():
        user = dataset.select(range(v[0], v[1]))
        output_dict[str(k)] = user
    dataset = DatasetDict(output_dict)
    return dataset


def create_train_test_split(dataset: datasets.Dataset | DatasetDict, test_split: float, col: str = None) -> datasets.DatasetDict:
    # TODO make split col variable
    if col is not None:
        dataset = create_train_test_split_by_column(dataset, test_split, col)
    elif isinstance(dataset, datasets.DatasetDict):
        dataset = create_train_test_split_by_keys(dataset, test_split)
    else:
        dataset = dataset.train_test_split(test_size=test_split)  # split data so there is a test split for eval
    return dataset


def load_train_test_split(path: str) -> Dict[str, List[str]]:
    with open(os.path.join(path, "train_test_split.json"), 'r') as f:
        split = json.load(f)
    return split


def create_train_test_split_by_column(data: Dataset, test_split: float, split_by_column: str):
    # splitting dataset by column
    log.info(f"Splitting by column {split_by_column}")
    all_options = data.unique(split_by_column)
    subset_size = max(1, int(len(all_options) * test_split))
    test_ids = random.sample(all_options, subset_size)
    test = data.filter(lambda example: example[split_by_column] in test_ids, num_proc=threads)
    train = data.filter(lambda example: example[split_by_column] not in test_ids, num_proc=threads)

    return DatasetDict({"train": train, "test": test})


def create_train_test_split_by_keys(data: DatasetDict, test_split: float) -> DatasetDict:
    # Right now the dictionary keys will be separated
    log.info(f"Splitting by keys")
    all_options = data.keys()
    subset_size = max(1, int(len(all_options) * test_split))
    test_ids = random.sample(all_options, subset_size)
    train = DatasetDict({k: v for k, v in data.items() if k not in test_ids})
    test = DatasetDict({k: v for k, v in data.items() if k in test_ids})
    # train = DatasetDict({'0': train['0'].select([0,1])})

    return DatasetDict({"train": train, "test": test})


def tokenize_data(data: Dataset, tokenizer: AutoTokenizer, test_split: float = .1) -> DatasetDict | Dataset:
    def preprocess_function(examples):
        # TODO examples may need to be changed here depended on dataset
        tokenized = tokenizer(examples["text"])
        return tokenized

    data = data.map(
        preprocess_function,
        batched=True,
        num_proc=threads,
        # remove_columns=data["train"].column_names,
    )

    return data


def get_data_and_tokenize(cfg, data_processor, tokenizer, split=.1):
    dataset = get_data_from_raw(cfg.data, cfg.data_dir)
    dataset, col_to_id_dict = preprocess_dataset(dataset, data_processor, cfg.tokenizer.numeric_bucket_amount)
    dataset = convert_to_huggingface(dataset, data_processor)
    tokenizer.add_special_tokens({'pad_token': '[PAD]', 'unk_token': '[UNK]'})
    log.info("DATASET PREPROCESSED, BEGINNING TOKENIZATION")
    dataset = tokenize_data(dataset, tokenizer)
    log.info("DATASET TOKENIZED")
    return dataset, col_to_id_dict


def save_dataset(dataset: Dataset, path: str):
    dataset.save_to_disk(path, num_proc=threads)


def to_train_dataloader(dataset: data.Dataset, collate_fn, batch_size: int) -> data.DataLoader:
    # todo distributed sampling?

    sampler = torch.utils.data.SequentialSampler(dataset)
    train_loader = data.DataLoader(dataset,
                                   batch_size=batch_size,
                                   sampler=sampler,
                                   num_workers=threads,
                                   drop_last=True,
                                   collate_fn=collate_fn
                                   )
    return train_loader


def to_val_dataloader(dataset: data.Dataset, collate_fn, batch_size: int) -> data.DataLoader:
    # todo distributed sampling?

    sampler = torch.utils.data.SequentialSampler(dataset)
    train_loader = data.DataLoader(dataset,
                                   batch_size=batch_size,
                                   sampler=sampler,
                                   num_workers=threads,
                                   drop_last=False,
                                   collate_fn=collate_fn
                                   )
    return train_loader


class NextTokenPredictionDataset(data.Dataset):
    """
    Returns a PyTorch Dataset object with labels extracted correctly for Causal Language
    Modeling (GPT-style next token prediction using a causal mask). Data must be a tensor of token ids.
    """

    def __init__(self, tokenized_dataset: Dataset, seq_length: int, pad_id: int, randomize_order: bool = False, fixed_cols: int = 2):
        self.pad_id = pad_id
        self.seq_length = seq_length  # number of ROWS in a sequence
        self.num_rows = sum([x for x in tokenized_dataset.num_rows.values()])
        self.user_ids = list(tokenized_dataset.keys())  # needs to be split by user already
        self.num_columns = len(tokenized_dataset[self.user_ids[0]]["input_ids"][0])
        self.data, self.label_mask = self.prepare_data(self.user_ids, tokenized_dataset)
        self.randomize_order = randomize_order
        self.fixed_cols = fixed_cols

    def __len__(self) -> int:
        return self.num_rows // self.seq_length  # consider end row

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        # The corresponding label for each example is a chunk of tokens of the same size,
        # but shifted one token to the right.

        # TODO: It is a arbitrary design choice whether to do the shift of the labels here
        # todo split by user?
        # or in the loss function. Huggingface's DataCollator is designed to let the loss
        # function do the shifting, so we need to change this if we want to be compatible with that.
        # start = i * self.seq_length
        # x = self.data[start: start + self.seq_length]
        # y = self.data[start + 1: start + self.seq_length + 1]
        # mask = self.label_mask[start: start + self.seq_length]
        # return {"input_ids": x, "targets": y, "mask": mask}
        x = self.data[i, :]
        # y = self.data[i, 1:]
        mask = self.label_mask[i, :]
        if self.randomize_order:
            index = torch.arange(x.shape[0])
            for s in range(self.seq_length):
                start = self.num_columns * s
                index[start:start + self.num_columns - self.fixed_cols] = torch.randperm(self.num_columns - self.fixed_cols) + start
            x = x[index]
            mask = mask[index]
        return {"input_ids": x, "mask": mask}

    def prepare_data(self, user_ids: List[str], tokenized_dataset: Dataset):
        pad_row = [self.pad_id for _ in range(self.num_columns)]  # this is a dummy pad row to add when number of rows is not multiple of sequence length
        all_sequences = []
        all_labels = []
        total_transactions = 0
        for uid in user_ids:
            # user_rows = tokenized_dataset.filter(lambda example: example["User"] == uid, num_proc=threads)  # todo this is very slow, better to have the data already giving right rows?
            # user_rows = torch.tensor(user_rows["input_ids"], dtype=torch.long)
            user_rows = torch.tensor(tokenized_dataset[uid]["input_ids"], dtype=torch.long)
            num_rows = user_rows.shape[0]
            total_transactions += (num_rows // self.seq_length) + int((num_rows % self.seq_length) > 0)  # sanity check

            # self.randomize_order = True
            # if self.randomize_order:
            #     index = torch.zeros_like(user_rows)
            #     index[:, -1] = self.num_columns-1
            #     for x in range(num_rows):
            #         index[x, :self.num_columns-1] = torch.randperm(self.num_columns-1)
            # else:
            #     index = torch.arange(self.num_columns).unsqueeze(0).repeat(num_rows, 1)
            index = torch.arange(self.num_columns).unsqueeze(0).repeat(num_rows, 1)

            user_rows = user_rows.gather(1, index)
            last_col_mask = torch.zeros_like(user_rows)
            last_col_mask[index == self.num_columns-2] = 1  # second to last column is where we will do the mask

            if num_rows % self.seq_length != 0:
                num_pad_rows = self.seq_length - (num_rows % self.seq_length)
                pad_rows = torch.tensor(pad_row).repeat([num_pad_rows, 1])
                user_rows = torch.cat([user_rows, pad_rows])
                last_col_mask = torch.cat([last_col_mask, torch.zeros_like(pad_rows)])
            assert len(user_rows) % self.seq_length == 0
            # if num_rows_for_user < self.seq_length:  # not enough transactions for the user to fill up the seq_len window
            #     continue
            for starting_row_id in range(0, len(user_rows) - self.seq_length + 1, self.seq_length):  # consider not striding by seq_len
                # add bos, eos tokens?
                seq = user_rows[starting_row_id: starting_row_id + self.seq_length].reshape(1, -1)
                all_sequences.append(seq)
                labels_for_sequence = last_col_mask[starting_row_id: starting_row_id + self.seq_length].reshape(1, -1)
                all_labels.append(labels_for_sequence)

        # note that with this implementation can have multiple users in same batch but NOT in same sequence (obviously)
        return torch.cat(all_sequences), torch.cat(all_labels)


class MaskedLanguageModelingDataset(data.Dataset):
    """
    Returns a PyTorch Dataset object with labels extracted correctly for Causal Language
    Modeling (GPT-style next token prediction using a causal mask). Data must be a tensor of token ids.
    """

    def __init__(self, data: torch.Tensor, n_cols: int):
        self.data = data
        self.n_cols = n_cols

    def __len__(self) -> int:
        return len(self.data) // self.n_cols

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # todo
        return x, y
