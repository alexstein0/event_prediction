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

log = logging.getLogger(__name__)

# THIS CODE IS COPIED FROM FATA-TRANS  This is the base class for masked collator when no temporal component and no static/dynamic split
class TransDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def __call__(self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        batch = _torch_collate_batch(examples, self.tokenizer)
        sz = batch.shape
        if self.mlm:
            batch = batch.view(sz[0], -1)
            inputs, labels = self.mask_tokens(batch)
            # print("MLM label shape: ", labels.view(sz).shape)
            # print("MLM batch shape: ", inputs.view(sz).shape)
            return {"input_ids": inputs.view(sz), "masked_lm_labels": labels.view(sz)}
        else:
            labels = batch.clone().detach()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            return {"input_ids": batch, "labels": labels}

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


def prepare_dataloaders(tokenized_dataset: DatasetDict, tokenizer, cfg: DictConfig) -> Dict[str, data.DataLoader]:
    """
    Takes in pretokenized hf dataset
    """

    # train_loader = data.DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True)
    # val_loader = data.DataLoader(val_data, batch_size=cfg.batch_size)
    train_data = tokenized_dataset["train"]
    val_data = tokenized_dataset["train"]
    train_loader = prepare_pretraining_dataloader(train_data, tokenizer, cfg)
    # val_loader = prepare_validation_dataloader(val_data, tokenizer)
    val_loader = prepare_pretraining_dataloader(val_data, tokenizer, cfg)
    return {"train": train_loader, "val": val_loader}


def prepare_pretraining_dataloader(tokenized_dataset: Dataset, tokenizer, cfg) -> data.DataLoader:

    if cfg.model.training_objective == "causal":
        tokenized_dataset = NextTokenPredictionDataset(tokenized_dataset, cfg.model.context_length, tokenizer.pad_token_id)
        collate_fn = FastDataCollatorForLanguageModeling(tokenizer=tokenizer, pad_to_multiple_of=cfg.impl.pad_to_multiple_of, mlm=False)

    elif cfg.model.training_objective == "masked":
        tokenized_dataset = MaskedLanguageModelingDataset(tokenized_dataset, n_cols)
        collate_fn = FastDataCollatorForLanguageModeling(tokenizer=tokenizer, pad_to_multiple_of=cfg.impl.pad_to_multiple_of, mlm=True)
    else:
        raise ValueError(f"training_objective must be 'causal' or 'masked', not {cfg.training_objective}")

    loader = to_dataloader(tokenized_dataset, collate_fn, cfg.model.batch_size)
    return loader


def preprocess_dataset(dataset, data_processor, numeric_bucket_amount: int = 5) -> datasets.Dataset:
    dataset = data_processor.normalize_data(dataset)
    for col in data_processor.get_numeric_columns():
        dataset[col], buckets = convert_to_binary_string(dataset[col], numeric_bucket_amount)

    # row_key = "NEW_ROW"
    row_token = "[ROW]"
    # dataset[row_key] = row_token
    #
    col_id = 0
    # todo right here is where we would extract the labels as well.
    #  Here we prepare string to be tokenized and below we will create string to tokenize
    all_cols = data_processor.get_data_cols()
    # all_cols.append(row_key)
    for col in all_cols:
        dataset[col] = str(col_id) + "_" + dataset[col].astype(str)
        col_id += 1

    dataset = Dataset.from_pandas(dataset)

    # dataset = dataset.map(lambda example: example, batched=True)
    threads = utils.get_cpus()

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


def split_data_by_columns(data: Dataset, test_split: float, split_by_column: str):
    # todo take in multiple columns (like user and card)
    all_options = data.unique(split_by_column)
    subset_size = int(len(all_options) * test_split)
    test_ids = random.sample(all_options, subset_size)
    test = data.filter(lambda example: example[split_by_column] in test_ids)
    train = data.filter(lambda example: example[split_by_column] not in test_ids)

    return DatasetDict({"train": train, "test": test})


def tokenize_data(data: Dataset, tokenizer: AutoTokenizer, test_split: float = .1) -> DatasetDict | Dataset:
    def preprocess_function(examples):
        # TODO examples may need to be changed here depended on dataset
        tokenized = tokenizer(examples["text"])
        return tokenized

    if test_split > 0:
        # TODO make split col variable
        col = "User"
        log.info(f"splitting by {col}")
        if col is not None:
            data = split_data_by_columns(data, test_split, col)
        else:
            data = data.train_test_split(test_size=test_split)  # split data so there is a test split for eval

    threads = utils.get_cpus()

    data = data.map(
        preprocess_function,
        batched=True,
        num_proc=threads,
        # remove_columns=data["train"].column_names,
    )

    return data


def update_attention_mask():
    pass

def get_data_and_tokenize(cfg, data_processor, tokenizer, split=.1):
    dataset = get_data_from_raw(cfg.data, cfg.data_dir, False, False)
    dataset = preprocess_dataset(dataset, data_processor, cfg.tokenizer.numeric_bucket_amount)
    tokenizer.add_special_tokens({'pad_token': '[PAD]', 'unk_token': '[UNK]'})
    log.info("DATASET PREPROCESSED, BEGINNING TOKENIZATION")
    dataset = tokenize_data(dataset, tokenizer)
    log.info("DATASET TOKENIZED")
    return dataset


def to_dataloader(dataset: data.Dataset, collate_fn, batch_size: int) -> data.DataLoader:
    # todo distributed sampling?

    sampler = torch.utils.data.SequentialSampler(dataset)
    train_loader = data.DataLoader(dataset,
                                   batch_size=batch_size,
                                   sampler=sampler,
                                   num_workers=utils.get_cpus(),
                                   drop_last=True,
                                   # collate_fn=collatefn
                                   )
    return train_loader


class NextTokenPredictionDataset(data.Dataset):
    """
    Returns a PyTorch Dataset object with labels extracted correctly for Causal Language
    Modeling (GPT-style next token prediction using a causal mask). Data must be a tensor of token ids.
    """

    def __init__(self, tokenized_dataset: Dataset, context_length: int, pad_id: int):
        data = torch.tensor(tokenized_dataset["input_ids"], dtype=torch.long)
        # last_col_mask = torch.tensor(tokenized_dataset["attention_mask"], dtype=torch.long)
        last_col_mask = torch.zeros_like(data)
        last_col_mask[:, -3] = 1  # second to last column is where we will do the attention mask

        self.flattened_data = data.reshape(-1)
        self.last_col_mask = last_col_mask.reshape(-1)
        self.context_length = context_length
        # self.flattened_data = torch.cat([self.flattened_data, torch.tensor([pad_id])])

    def __len__(self) -> int:
        return len(self.flattened_data) // self.context_length

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        # The corresponding label for each example is a chunk of tokens of the same size,
        # but shifted one token to the right.

        # TODO: It is a arbitrary design choice whether to do the shift of the labels here
        # or in the loss function. Huggingface's DataCollator is designed to let the loss
        # function do the shifting, so we need to change this if we want to be compatible with that.
        start = i * self.context_length
        x = self.flattened_data[start: start + self.context_length]
        y = self.flattened_data[start + 1: start + self.context_length + 1]
        mask = self.last_col_mask[start: start + self.context_length]
        return {"input_ids": x, "targets": y, "mask": mask}


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
