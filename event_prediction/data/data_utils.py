import io
import json
import logging
import os
import tarfile
from typing import Dict, List, Tuple, Union
from urllib.parse import urlparse

import datasets
import numpy as np
import pandas as pd
import requests
import torch
from datasets import Dataset
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from torch.utils import data
from tqdm import tqdm

log = logging.getLogger(__name__)


def get_data_from_raw(cfg, raw_data_dir_name="data_raw", save_tar_to_disk=False, save_csv_to_disk=False) -> pd.DataFrame:
    """
    Return a dataframe of the dataset specified in the config. For a given dataset
    first we will look for it on disk, and if it is not there we will download it.
    Handles either .csv file or .tgz file with single .csv file inside.
    """
    data_dir = os.path.join(get_original_cwd(), raw_data_dir_name)
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
        bytes = extract(bytes)     
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
    X = X.astype(str)
    return X

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


def normalize_numeric(df: pd.DataFrame, normalize_type: str) -> pd.DataFrame:
    # todo add other types of normalization
    if normalize_type == "normal":
        df = (df - df.mean(0)) / df.std(0)
    else:
        log.info("No normalization applied")
    return df

def concat_dataframe_cols(df: pd.DataFrame) -> pd.Series:
    return df.astype(str).apply('_'.join, axis=1)

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


def add_index_tokens(dataset: pd.DataFrame, labels: pd.DataFrame) -> (pd.DataFrame, List[str]):
    special_tokens_added = []
    index_tokens = []
    dataset.reset_index(drop=True, inplace=True)
    labels.reset_index(drop=True, inplace=True)
    for token in labels.columns:
        tok_locs = labels[token][labels[token] != labels[token].shift()].copy().astype(str)
        tok_locs[:] = token
        tok_locs = tok_locs.to_frame()
        tok_locs.columns = ["spec"]
        index_tokens.append(tok_locs)
        special_tokens_added.append(token)
    combined = pd.concat([*index_tokens, dataset], axis=0).sort_index().reset_index(drop=True)
    return combined, special_tokens_added

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

#
# def add_train_transacations_to_testset(
#         trainset: pd.DataFrame,
#         testset: pd.DataFrame,
#         train_test_users: set,
#         sample_size: int = 10,
#         consider_card: bool = False,
# ) -> pd.DataFrame:
#     """
#     Add a sampling of transactions from the trainset for each trainset user that also
#     appears in the testset.
#     """
#     groupby_columns = ["User", "Card"] if consider_card else ["User"]
#     sort_columns = (
#         ["User", "Card", "total_minutes"]
#         if consider_card
#         else ["User", "total_minutes"]
#     )
#     # Get the indices of the last y-1 transactions for each user in dataframe x
#     get_sample_indices = lambda x, y: x.index[-(y - 1):]
#
#     test_extra_indices = (
#         trainset.loc[trainset["User"].isin(train_test_users)]
#             .groupby(groupby_columns)
#             .apply(get_sample_indices, sample_size)
#     )
#     test_extra_indices = test_extra_indices.explode()
#     testset = pd.concat([trainset.loc[test_extra_indices], testset])
#     testset.sort_values(by=sort_columns, inplace=True)
#     return testset


def add_static_user_fields(
        trainset: pd.DataFrame,
        testset: pd.DataFrame,
        test_only_users: set,
        consider_card: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add "static field" columns to the dataset. This is aggregate data that is added
    to every transaction for a given user.
    The columns to add are:
    1. Average dollar amount of the user
    2. Standard deviation dollar amount of the user
    3. Most frequent MCC
    4. Most frequent Use Chip
    """
    get_top_item = lambda x: x.mode().iloc[0]

    # One row per train-set user
    groupby_columns = ["User", "Card"] if consider_card else ["User"]
    sort_columns = (
        ["User", "Card", "total_minutes"]
        if consider_card
        else ["User", "total_minutes"]
    )
    train_static_data = trainset.groupby(groupby_columns).agg(
        avg_dollar_amt=("Amount", "mean"),
        std_dollar_amt=("Amount", "std"),
        top_mcc=("MCC", get_top_item),
        top_chip=("Use Chip", get_top_item),
    )

    # Gather dataset-level values for use in filling NaNs for individual users
    dataset_amt_avg = trainset["Amount"].mean()
    dataset_amt_std = trainset["Amount"].std()
    dataset_top_mcc = trainset["MCC"].mode().iloc[0]
    dataset_top_chip = trainset["Use Chip"].mode().iloc[0]
    dataset_static_values = {
        "avg_dollar_amt": dataset_amt_avg,
        "std_dollar_amt": dataset_amt_std,
        "top_mcc": dataset_top_mcc,
        "top_chip": dataset_top_chip,
    }

    # Replace NaNs
    train_static_data.fillna(value=dataset_static_values, inplace=True)

    # Add static data columns to trainset
    trainset = trainset.join(train_static_data, on="User")

    # For testset, operate on the principle that at inference time we can lookup user-level
    # static data from some previous transactions. For testset users that are not in the
    # trainset (i.e. new users with no previous transactions), we can use the dataset-level
    # values. Here we create a dataframe with those dataset-level values repeated enough
    # times to cover every row with a not-in-trainset user.
    test_only_user_indices = list(test_only_users)
    dataset_values_for_testset = pd.DataFrame(
        {
            "avg_dollar_amt": np.repeat(dataset_amt_avg, len(test_only_users)),
            "std_dollar_amt": np.repeat(dataset_amt_std, len(test_only_users)),
            "top_mcc": np.repeat(dataset_top_mcc, len(test_only_users)),
            "top_chip": np.repeat(dataset_top_chip, len(test_only_users)),
        },
        index=test_only_user_indices,
    )

    # Now we have static data for every user in the trainset and all all the additional
    # testset users not apearing in the trainset. If we  concat this we have static data
    # for all users that might possibly appear in the testset (a superset of those users).
    test_static_data = pd.concat([train_static_data, dataset_values_for_testset])

    # Add static data columns to trainset
    testset = testset.join(test_static_data, on="User")

    trainset.sort_values(by=sort_columns, inplace=True)
    testset.sort_values(by=sort_columns, inplace=True)

    return trainset, testset


def save_processed_dataset(dataset: List[str], cfg, processed_data_dir_name="data", sep='\n') -> str:
    """
    Save list of strings to a text file. They are saved with a newline seperator between
    each string in the list by default.
    """
    data_dir = os.path.join(get_original_cwd(), processed_data_dir_name)
    filepath = os.path.join(data_dir, f"{cfg.name}.txt")
    os.makedirs(data_dir, exist_ok=True)

    if sep is not None:
        dataset = [line + sep for line in dataset]

    with open(filepath, "w") as f:
        f.writelines(dataset)
    return filepath


def load_processed_dataset(cfg, processed_data_dir_name="data", sep='\n') -> List[str]:
    """
    Load the contents of a text file to a single string, with no newlines or 
    whitespace removed.
    """
    data_dir = os.path.join(get_original_cwd(), processed_data_dir_name)
    filepath = os.path.join(data_dir, f"{cfg.name}.txt")
    with open(filepath, "r") as f:
        dataset = f.read()
    
    dataset = dataset.strip().split(sep)
    return dataset


def save_json(data: Dict, file_dir: str, file_name: str) -> str:
    file_dir = os.path.join(get_original_cwd(), file_dir)
    filepath = os.path.join(file_dir, file_name)
    os.makedirs(file_dir, exist_ok=True)

    with open(filepath, "w") as outfile:
        json.dump(data, outfile)
    return filepath

def read_json(file_dir: str, file_name: str) -> Dict:
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


def get_dataloader(cfg: DictConfig, tokenizer, str_tokens):
    """
    Takes a string of raw text tokens and a tokenizer for encoding and returns a PyTorch 
    Dataloader object with examples, labels batches ready for training.
    The labels can be either the next token in the sequence (causal language modeling)
    or a masked token (masked language modeling).
    """
    id_tokens = tokenizer.encode(str_tokens)
    
    if cfg.training_objective == "causal":
        dataset = NextTokenPredictionDataset(id_tokens, cfg.context_length)
    elif cfg.training_objective == "masked":
        raise NotImplementedError
    else:
        raise ValueError(f"training_objective must be 'causal' or 'masked', not {cfg.training_objective}")
    
    train_loader, val_loader = to_dataloader(cfg, dataset)
    return train_loader, val_loader
    

def to_dataloader(cfg: DictConfig, dataset: data.Dataset) -> Tuple[data.DataLoader, data.DataLoader]:
    """Given a PyTorch Dataset and config for batch size, return a Pytorch DataLoader."""
    n = len(dataset)
    train_size = int(cfg.train_ratio * n)
    val_size = n - train_size
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