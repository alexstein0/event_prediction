import io
import json
import logging
import os
import tarfile
from typing import Dict, List, Tuple, Union
from urllib.parse import urlparse

import datasets
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer
import multiprocessing

import numpy as np
import pandas as pd
import requests
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from torch.utils import data
from tqdm import tqdm
from datasets import Dataset, load_dataset

log = logging.getLogger(__name__)

def get_huggingface_dataset(cfg):
    data_files = {"train": cfg.url}
    dataset = load_dataset("csv", data_files=data_files)
    return dataset['train']


def get_data_from_raw(cfg, raw_data_dir_name="data_raw", save_tar_to_disk=False, save_csv_to_disk=False) -> pd.DataFrame:
    """
    Return a dataframe of the dataset specified in the config. For a given dataset
    first we will look for it on disk, and if it is not there we will download it.
    Handles either .csv file or .tgz file with single .csv file inside.
    """
    data_dir = os.path.join(get_original_cwd(), raw_data_dir_name)
    csv_file = os.path.join(data_dir, f"{cfg.name}.csv")
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    else:
        _, ext = os.path.splitext(urlparse(cfg.url).path)
        filepath = os.path.join(data_dir, f"{cfg.name}{ext}")
        if os.path.exists(filepath):
            bytes = read_bytes(filepath)
        else:
            bytes = download_data_from_url(cfg.url)
        if ext == ".tgz":
            if save_tar_to_disk:
                os.makedirs(data_dir, exist_ok=True)
                write_bytes(bytes, filepath)
            try:    
                bytes = extract(bytes)
            except tarfile.ReadError as e:
                log.error(f"Error when trying to extract file. Double-check that the URL actually exists: {cfg.url}")
                raise     
        df = pd.read_csv(bytes)
        if save_csv_to_disk:
            os.makedirs(data_dir, exist_ok=True)
            filepath = os.path.join(data_dir, f"{cfg.name}.csv")
            df.to_csv(filepath, index=False)
    return df


def download_data_from_url(url: str) -> io.BytesIO:
    """Download a file to memory without writing it to disk"""
    response = requests.get(url, stream=True)
    file_size = int(response.headers.get("Content-Length", 0))
    # Initialize a downloader with a progress bar
    downloader = TqdmToLogger(
        log,
        iterable=response.iter_content(1024),
        desc=f"Downloading {url}",
        total=file_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    )
    data = io.BytesIO()
    for chunk in downloader.iterable:
        data.write(chunk)
        # Update the progress bar manually
        downloader.update(len(chunk))
    # Reset the file object position to the start of the stream
    data.seek(0)
    return data


def write_bytes(data: io.BytesIO, filepath: str) -> None:
    log.info(f"Saving to {filepath}")
    with open(filepath, "wb") as f:
        for byte in data:
            f.write(byte)
    # Reset the file object position to the start of the stream
    data.seek(0)


def read_bytes(filepath: str) -> io.BytesIO:
    with open(filepath, "rb") as file:
        content = file.read()
    return io.BytesIO(content)


def extract(data: io.BytesIO) -> io.BytesIO:
    """Extract a tar.gz file to memory"""
    log.info(f"Extracting tar.gz file...")
    with tarfile.open(fileobj=data, mode='r:gz') as tar:
        num_files = len(tar.getmembers())
        assert num_files == 1, f"Expected single csv file in tarball but got {num_files} files."
        for member in tar.getmembers():
            f = tar.extractfile(member)
            if f is not None:
                data = f.read()
    return io.BytesIO(data)


def get_timestamps(X: pd.DataFrame,
                   year_col: str="Year",
                   month_col: str="Month",
                   day_col: str="Day",
                   time_col: str="Time") -> pd.Series:
    """Return a pd.Series of datetime objects created from a dataframe with columns 'Year', 'Month', 'Day', 'Time'"""
    X_hm = X[time_col].str.split(
        ":", expand=True
    )  # Expect "Time" to be in the format "HH:MM"
    d = pd.to_datetime(
        dict(
            year=X[year_col], month=X[month_col], day=X[day_col], hour=X_hm[0], minute=X_hm[1]
        )
    )
    return d


def add_hours_total_minutes(X: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe with new columns 'Hour' and 'total_minutes'"""
    timestamps = get_timestamps(X)
    X["Hour"] = timestamps.dt.hour
    # Add a column for total minutes from timestamp=0 to our dataframe
    zero_time = pd.to_datetime(np.zeros(len(X)))
    total_seconds = (timestamps - zero_time).dt.total_seconds().astype(int)
    total_minutes = total_seconds // 60
    X["total_minutes"] = total_minutes
    return X


def add_is_online(X: pd.Series, flag: str="ONLINE") -> pd.Series:
     return X == flag


def add_minutes_from_last(X: pd.DataFrame, minutes_col: str, by_columns: List[str] = None) -> pd.DataFrame:
    if by_columns is not None:
        col = X.groupby(by_columns)[minutes_col]
    else:
        col = X[minutes_col].copy()
    col = col.diff().fillna(0).astype("int64")
    X["total_minutes_from_last"] = col
    return X

def convert_to_str(X: pd.Series) -> pd.Series:
    X = X.convert_dtypes(convert_integer=True)
    null_spots = X.isna()
    X = X.astype(str)
    X[null_spots] = "NAN"
    return X

def convert_to_bool(X: pd.Series) -> pd.Series:
    rep = {'yes': True,
           'no': False,
           'true': True,
           'false': False}
    X = X.str.lower()
    X = X.replace(rep)
    return X.astype('bool')

def convert_dollars_to_floats(X: pd.Series, log_scale: bool = True) -> pd.Series:
    X = X.str.replace("$", "").astype(float)
    if log_scale:
        X = np.log(X)
    return X


def bucket_numeric(X: pd.Series, bin_type: str, num_bins: Union[int, List[float]]) -> (pd.Series, pd.array):
    """
    Convert all numeric values to integers based on a specified number of bins.
    "uniform" bins will be of equal size, "quantile" bins will have an equal number of
    values in each bin.
    """
    assert bin_type in ["uniform", "quantile"], f"bin_type must be 'uniform' or 'quantile', not {bin_type}"

    if bin_type == "uniform":
        out, bins = pd.cut(X, bins=num_bins, retbins=True, labels=False, duplicates='drop')
    elif bin_type == "quantile":
        out, bins = pd.qcut(X, q=num_bins, retbins=True, labels=False, duplicates='drop')
    else:
        out, bins = None, None  # todo
    return out, bins


def convert_to_binary_string(X: pd.Series, digits_remaining: int = -1) -> (pd.Series, pd.array):
    if len(X) == 0:
        return [], pd.Series()  # shouldnt get here
    if len(X.unique()) == 1:
        return [''], X  # need to account for duplicates
    if digits_remaining == 0:
        return [''], pd.Series()  # all tokens in this bucket will be the same
    med = X.median()
    if med == X.max():
        return [''], pd.Series()  # because we split <= sometimes everything is in the first bucket (this might solve uniqueness problem too)
    left = X <= med  # gets a 0
    right = X > med  # gets a 1
    digits_remaining = max(digits_remaining - 1, -1)
    left_strings, left_buckets = convert_to_binary_string(X[left], digits_remaining)
    right_strings, right_buckets = convert_to_binary_string(X[right], digits_remaining)
    left_tokens = ['0' + x for x in left_strings]
    right_tokens = ['1' + x for x in right_strings]
    # output = pd.Series()
    output_series = pd.Series(index=range(len(X))).astype(str)
    output_series[left.reset_index(drop=True)] = left_tokens
    output_series[right.reset_index(drop=True)] = right_tokens
    return output_series, pd.concat([left_buckets, right_buckets], ignore_index=True)



def normalize_numeric(df: pd.DataFrame, normalize_type: str) -> pd.DataFrame:
    # todo add other types of normalization
    if normalize_type == "normal":
        df = (df - df.mean(0)) / df.std(0)
    else:
        log.info("No normalization applied")
    return df

def concat_dataframe_cols(df: pd.DataFrame, separator: str= "_") -> pd.Series:
    return df.astype(str).apply(separator.join, axis=1)

def add_special_tabular_tokens(df: pd.DataFrame, add_col_sep: str='COL', add_row_sep: str='ROW') -> pd.DataFrame:
    output = pd.DataFrame()
    if add_col_sep is not None:
        for col in df.columns:
            output[col] = df[col]
            output[f'sep_{col}'] = add_col_sep
        output.drop(output.columns[-1], axis=1)
    if add_row_sep is not None:
        output['row_sep'] = add_row_sep

    return output

def cols_to_words(df: pd.DataFrame, second_table: pd.DataFrame=None) -> pd.Series:
    df["index_col"] = df.index
    all_tokens = df.values.tolist()
    if second_table is not None:
        second_table["index_col"] = second_table.index
        second_tokens = second_table.values.tolist()
        all_tokens.extend(second_tokens)
    all_tokens.sort(key=lambda x: x[-1])
    all_tokens = [x for row in all_tokens for x in row[:-1]]
    return pd.Series(all_tokens)

def remove_spaces(X: pd.Series) -> pd.Series:
    return X.str.replace(' ', '')


def get_prepended_tokens(labels: pd.DataFrame) -> (pd.DataFrame, List[str]):
    special_tokens_added = []
    index_tokens = []
    for token in labels.columns:
        tok_locs = labels[token][labels[token] != labels[token].shift()].copy().astype(str)
        tok_locs[:] = token
        tok_locs = tok_locs.to_frame()
        tok_locs.columns = ["prepended_tokens"]
        index_tokens.append(tok_locs)
        special_tokens_added.append(token)
    combined = pd.concat(index_tokens, axis=0)
    combined = combined.groupby(combined.index)["prepended_tokens"].apply(list)
    return combined, special_tokens_added


def interweave_series(datasets: List[pd.Series]) -> (pd.DataFrame):
    for ds in datasets:
        ds.reset_index(drop=True, inplace=True)
        ds.name = ["tokens"]

    return pd.concat([datasets], axis=0).sort_index().reset_index(drop=True)


# def get_train_test_split(X: pd.DataFrame, split_year: int = 2018) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     """Return a train-test split of the data based on a single year cutoff"""
#     train = X.loc[X["Year"] < split_year]
#     test = X.loc[X["Year"] >= split_year]
#     return train, test


def get_users(trainset: pd.DataFrame, testset: pd.DataFrame, user_col: str="User") -> Tuple[set, set]:
    """Return a list of users in both train and test sets, and users in both."""
    train_users = set(trainset[user_col].unique())
    test_users = set(testset[user_col].unique())
    train_test_users = train_users.intersection(test_users)
    test_only_users = test_users.difference(train_users)
    return train_test_users, test_only_users


def concatenated_col(df: pd.DataFrame, cols_to_concat: List[str]) -> pd.Series:
    """Create a Series (single column) that is a concatenation of selected columns in a df."""
    return df[cols_to_concat].astype(str).apply('_'.join, axis=1)


def add_static_fields(df: pd.DataFrame, reference_df: pd.DataFrame=None, groupby_columns=["User", "Card"]) -> pd.DataFrame:
    # reference_df is historic data that can be accessed at inference time. At train time, we can we can simply
    # use reuse the trainset as the reference dataset so that we add static values for all users in the trainset.
    user_static_values = get_user_level_static_values(reference_df, groupby_columns)
    
    # Add the static values for the users that appeared in the reference dataset. Since we are using a left join, this will
    # add NaNs for users that did not appear in the reference dataset. (Users that appeared in the reference dataset but not
    # in the current dataset are ignored by left join.)
    df = df.merge(user_static_values, on=groupby_columns, how="left")
    
    # Fill in missing user-level values with dataset-level values
    dataset_static_values = get_dataset_level_static_values(reference_df)
    df.fillna(value=dataset_static_values, inplace=True)

    return df
    

def get_user_level_static_values(df: pd.DataFrame, groupby_columns=["User", "Card"]) -> pd.DataFrame:
    """
    Computes static values from a DataFrame aggregated by the groupby columns (e.g. ["User", "Card"]).
    The columns to add are:
    1. Average dollar amount of the user
    2. Standard deviation dollar amount of the user
    3. Most frequent MCC
    4. Most frequent Use Chip
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        groupby_columns (List[str]): The columns to group by.
    Returns:
        pd.DataFrame: A DataFrame containing the static values, with one row per groupby combination. (e.g. one row per user)
    """

    assert pd.api.types.is_numeric_dtype(df['Amount']), f"Expected 'Amount' col to have numeric dtype but got: {df['Amount'].dtype}"
    get_most_frequent_item = lambda x: x.mode().iloc[0]
    grouped_static_df = df.groupby(groupby_columns).agg(
        avg_dollar_amt=("Amount", "mean"),
        std_dollar_amt=("Amount", "std"),
        top_mcc=("MCC", get_most_frequent_item),
        top_chip=("Use Chip", get_most_frequent_item),
    )
    return grouped_static_df


def get_dataset_level_static_values(df: pd.DataFrame) -> Dict:
    """Gather dataset-level values"""
    dataset_amt_avg = df["Amount"].mean()
    dataset_amt_std = df["Amount"].std()
    dataset_top_mcc = df["MCC"].mode().iloc[0]
    dataset_top_chip = df["Use Chip"].mode().iloc[0]
    dataset_static_values = {
        "avg_dollar_amt": dataset_amt_avg,
        "std_dollar_amt": dataset_amt_std,
        "top_mcc": dataset_top_mcc,
        "top_chip": dataset_top_chip,
    }
    return dataset_static_values


def create_dataset(df: pd.DataFrame) -> Dataset:
    dataset = Dataset.from_pandas(df)
    return dataset

def save_processed_dataset(dataset: Dataset, processed_data_dir_name: str, processed_data_file_name: str) -> str:
    """
    Save list of strings to a text file. They are saved with a newline seperator between
    each string in the list by default.
    the file is saved as a list of jsons where the keys are labels (such as 'text' and 'label')
    """
    data_dir = os.path.join(get_original_cwd(), processed_data_dir_name)
    filepath = os.path.join(data_dir, f"{processed_data_file_name}")
    os.makedirs(data_dir, exist_ok=True)
    # with open(filepath, "w") as outfile:
    #     json.dump(list_of_dicts, outfile)
    dataset.save_to_disk(filepath)

    # keys = dataset.keys()
    # values_list = zip(*(dataset[key] for key in keys))
    #
    # list_of_dicts = [dict(zip(keys, values)) for values in values_list]

    # return save_json(dataset, processed_data_dir_name, f"{processed_data_file_name}.json")
    return filepath


def load_processed_dataset(processed_data_dir_name: str, processed_data_file_name: str) -> pd.DataFrame:
    """
    Load the contents of a text file to a single string, with no newlines or 
    whitespace removed.
    """
    data_dir = os.path.join(get_original_cwd(), processed_data_dir_name)
    filepath = os.path.join(data_dir, f"{processed_data_file_name}")

    dataset = Dataset.load_from_disk(filepath)

    # list_of_dicts = read_json(processed_data_dir_name, f"{processed_data_file_name}.json")
    # # todo make this more generic
    # texts = []
    # labels = []
    # for i in list_of_dicts:
    #     texts.append(i['text'])
    #     labels.append(i['label'])
    # return {'text': texts, 'label': labels}
    return dataset.to_pandas()


def save_json(data: [Dict | List[Dict]], file_dir: str, file_name: str) -> str:

    file_dir = os.path.join(get_original_cwd(), file_dir)
    filepath = os.path.join(file_dir, file_name)
    os.makedirs(file_dir, exist_ok=True)

    with open(filepath, "w") as outfile:
        json.dump(data, outfile)
    return filepath


def read_json(file_dir: str, file_name: str) -> Dict | List[Dict]:

    filepath = os.path.join(get_original_cwd(), file_dir, file_name)
    assert os.path.exists(filepath), f"File not found at {filepath}"
    assert os.path.getsize(filepath) > 0, f"File is empty at {filepath}"

    with open(filepath, "r") as file:
        try:
            data = json.load(file)
        except json.decoder.JSONDecodeError as e:
            log.error(f"File {filepath} is not in JSON format: {e}")
            raise e
    return data


def get_dataloader(cfg: DictConfig, tokenizer, tokens: List[str] | List[int], is_ids: bool = True, split_point: int = None) -> Tuple[data.DataLoader, data.DataLoader]:
    """
    Takes a string of raw text tokens and a tokenizer for encoding and returns a PyTorch 
    Dataloader object with examples, labels batches ready for training.
    The labels can be either the next token in the sequence (causal language modeling)
    or a masked token (masked language modeling).
    """
    if not is_ids:
        id_tokens = tokens.map(lambda example: tokenizer(example["text"]), batched=True)["input_ids"]
    else:
        id_tokens = tokens

    id_tokens = torch.tensor(id_tokens, dtype=int).reshape(-1)

    # id_tokens = tokenizer.encode(str_tokens)
    
    if cfg.training_objective == "causal":
        dataset = NextTokenPredictionDataset(id_tokens, cfg.context_length)
    elif cfg.training_objective == "masked":
        raise NotImplementedError
    else:
        raise ValueError(f"training_objective must be 'causal' or 'masked', not {cfg.training_objective}")
    
    train_loader, val_loader = to_dataloader(cfg, dataset, split_point=split_point)
    return train_loader, val_loader


def preprocess_dataset(dataset, data_processor, numeric_bucket_amount: int = 5) -> datasets.Dataset:

    dataset = data_processor.normalize_data(dataset)
    for col in data_processor.get_numeric_columns():
        dataset[col], buckets = convert_to_binary_string(dataset[col], numeric_bucket_amount)

    col_id = 0
    for col in data_processor.get_all_cols():
        dataset[col] = str(col_id) + "_" + dataset[col].astype(str)
        col_id += 1

    dataset = Dataset.from_pandas(dataset)

    def concat_columns(example):
        new_ex = {}
        new_ex["text"] = " ".join(example.values())
        return new_ex

    dataset = dataset.map(lambda example: example, batched=True)
    try:
        threads = max(os.cpu_count(), multiprocessing.cpu_count(), 1)
    except:
        threads = 1
    dataset = dataset.map(concat_columns, num_proc=threads)
    dataset = dataset.select_columns("text")
    return dataset

def preprocess_and_tokenize_data(data: Dataset, tokenizer: AutoTokenizer, test_split: float=.0) -> DatasetDict|Dataset:
    # def preprocess_function(examples):
    #     return tokenizer([" ".join(x) for x in examples["text"]])
    def preprocess_function(examples):
        # TODO examples may need to be changed here depended on dataset
        tokenized = tokenizer(examples["text"])
        return tokenized

    if test_split > 0:
        print("splitting")
        data = data.train_test_split(test_size=test_split)  # split data so there is a test split for eval
    try:
        threads = max(os.cpu_count(), multiprocessing.cpu_count(), 1)
    except:
        threads = 1

    tokenized_data = data.map(
        preprocess_function,
        batched=True,
        num_proc=threads,
        # remove_columns=data["train"].column_names,
    )
    # flatted_train = [item for row in tokenized_data["train"]["input_ids"] for item in row]
    # flatted_test = [item for row in tokenized_data["test"]["input_ids"] for item in row]
    # return flatted_train, flatted_test
    return [item for row in tokenized_data["input_ids"] for item in row]
    # data = data.map(preprocess_function, num_proc=threads, batched=True)
    # return data
    # block_size = 128

    # def group_texts(examples):
    # I think this groups texts across attributes
    #     print(examples)
    #     # Concatenate all texts.
    #     concatenated_examples = {k: examples[k] for k in examples.keys()}
    #     total_length = len(concatenated_examples[list(examples.keys())[0]])
    #     # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    #     # customize this part to your needs.
    #     if total_length >= block_size:
    #         total_length = (total_length // block_size) * block_size
    #     # Split by chunks of block_size.
    #     result = {
    #         k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
    #         for k, t in concatenated_examples.items()
    #     }
    #     result["labels"] = result["input_ids"].copy()
    #     return result
    # tokenized_data = tokenized_data.map(group_texts, batched=True, num_proc=4)
    # return tokenized_data

def to_dataloader(cfg: DictConfig, dataset: data.Dataset, split_point: int = None) -> Tuple[data.DataLoader, data.DataLoader]:
    """Given a PyTorch Dataset and config for batch size, return a Pytorch DataLoader."""
    n = len(dataset)
    train_size = int(cfg.train_ratio * n)
    val_size = n - train_size
    if split_point is not None:
        train_data = data.Subset(dataset, range(0, split_point))
        val_data = data.Subset(dataset, range(split_point, len(dataset)))
    else:
        train_data, val_data = data.random_split(dataset, [train_size, val_size])
    train_loader = data.DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True)
    val_loader = data.DataLoader(val_data, batch_size=cfg.batch_size)
    return train_loader, val_loader


class NextTokenPredictionDataset(data.Dataset):
    """
    Returns a PyTorch Dataset object with labels extracted correctly for Causal Language 
    Modeling (GPT-style next token prediction using a causal mask). Data must be a tensor of token ids.
    """
    def __init__(self, data: torch.Tensor, context_length: int):
        self.data = data
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.data) // self.context_length

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # The corresponding label for each example is a chunk of tokens of the same size,
        # but shifted one token to the right.
        
        # TODO: It is a arbitrary design choice whether to do the shift of the labels here
        # or in the loss function. Huggingface's DataCollator is designed to let the loss
        # function do the shifting, so we need to change this if we want to be compatible with that.
        x = self.data[i : i + self.context_length]
        y = self.data[i + 1 : i + self.context_length + 1]
        return x, y

class TqdmToLogger(tqdm):
    """File-like object to redirect tqdm output to a logger."""
    def __init__(self, logger, level=logging.INFO, *args, **kwargs):
        self.logger = logger
        self.level = level
        super().__init__(*args, **kwargs)

    def write(self, s):
        # Only log if the message is not empty or just a newline
        if s.rstrip() != '':
            self.logger.log(self.level, s.rstrip())

    def flush(self):
        pass