import io
import logging
import os
import tarfile
from typing import Tuple, Union
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from hydra.utils import get_original_cwd
from tqdm import tqdm

log = logging.getLogger(__name__)

def load_dataset(cfg, data_dir_name):
    #TODO
    pass

def get_data_from_raw(cfg, raw_data_dir_name="data_raw", save_tar_to_disk=False, save_csv_to_disk=False) -> pd.DataFrame:
    """
    Return a dataframe of the dataset specified in the config. For a given dataset
    first we will look for it on disk, and if it is not there we will download it.
    """
    data_dir = os.path.join(get_original_cwd(), raw_data_dir_name)
    _, ext = os.path.splitext(urlparse(cfg.url).path)
    file_path = os.path.join(data_dir, cfg.name)
    try:
        # If we have the data on disk, load it
        data = get_data_from_file(file_path)
    except:
        os.makedirs(data_dir, exist_ok=True)
        data = download_data_from_url(cfg.url)
        if save_tar_to_disk:
            save_raw_data(data, f"{file_path}.tgz")
        if ext == ".tgz":
            data = extract(data)
        # pd.read_csv can take a filename or a file-like object, so we can directly pass our BytesIO data object.
        data = pd.read_csv(data)
        if save_csv_to_disk:
            data.to_csv(f"{file_path}.csv")
    return data


def extract(data: io.BytesIO) -> io.BytesIO:
    """Extract a tar.gz file to memory"""
    log.info(f"Extracting tar.gz file...")
    with tarfile.open(fileobj=data, mode='r:gz') as tar:
        for member in tar.getmembers():
            f = tar.extractfile(member)
            if f is not None:
                data = f.read()
    return io.BytesIO(data)


def download_data_from_url(url: str) -> io.BytesIO:
    try:
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

    except Exception as e:
        log.error(f"Error: {e}")
    return data


def save_raw_data(data, filepath):
    log.info(f"Saving to {filepath}")
    with open(filepath, "wb") as f:
        for byte in data:
            f.write(byte)
    # Reset the file object position to the start of the stream
    data.seek(0)


def get_data_from_binary(filepath: str):
    with tarfile.open(filepath, "r:gz") as tar:
        for member in tar.getmembers():
            f = tar.extractfile(member)
            if f is not None:
                content = f.read()
    return content

def get_data_from_file(filepath: str) -> pd.DataFrame:
    log.info(f"Checking {filepath} to load data...")
    try:
        # read csv
        data = pd.read_csv(f"{filepath}.csv")
    except:
        try:
            # read binary file
            with tarfile.open(filepath, "r:gz") as tar:
                for member in tar.getmembers():
                    f = tar.extractfile(member)
                    if f is not None:
                        data = f.read()
                        data = pd.read_csv(io.BytesIO(data))
                        break
        except:
            raise ValueError(
                f"input must be filename of type byte or csv"
            )
    log.info(f"Data load complete")
    return data


def get_timestamps(X: pd.DataFrame) -> pd.Series:
    """Return a pd.Series of datetime objects created from a dataframe with columns 'Year', 'Month', 'Day', 'Time'"""
    X_hm = X["Time"].str.split(
        ":", expand=True
    )  # Expect "Time" to be in the format "HH:MM"
    d = pd.to_datetime(
        dict(
            year=X["Year"], month=X["Month"], day=X["Day"], hour=X_hm[0], minute=X_hm[1]
        )
    )
    return d


def add_hours_total_minutes(X: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe with new columns 'Hour' and 'total_minutes'"""
    timestamps = get_timestamps(X)
    X["Hour"] = timestamps.dt.hour
    # Add a column for total minutes from timestamp=0 to our datafrae
    zero_time = pd.to_datetime(np.zeros(len(X)))
    total_seconds = (timestamps - zero_time).dt.total_seconds().astype(int)
    total_minutes = total_seconds // 60
    X["total_minutes"] = total_minutes
    return X


def convert_dollars_to_floats(X: pd.DataFrame, log_scale: bool = True) -> pd.DataFrame:
    """Return a dataframe with the 'Amount' column converted to floats"""
    X["Amount"] = X["Amount"].str.replace("$", "").astype(float)
    if log_scale:
        X["Amount"] = np.log(X["Amount"])
    return X

def do_basic_preprocessing(X: pd.DataFrame, consider_card: bool = False) -> pd.DataFrame:
    # todo is it faster to process each row or to try it this way?
    """Return a preprocessed dataframe"""
    X = add_hours_total_minutes(X)
    X = convert_dollars_to_floats(X, log_scale=True)
    sort_columns = (
        ["User", "Card", "total_minutes"]
        if consider_card
        else ["User", "total_minutes"]
    )
    X = X.sort_values(by=sort_columns)
    # Add a column numbering the transactions in order
    X["rownumber"] = np.arange(len(X))
    return X


def get_train_test_split(X: pd.DataFrame, split_year: int = 2018) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return a train-test split of the data based on a single year cutoff"""
    train = X.loc[X["Year"] < split_year]
    test = X.loc[X["Year"] >= split_year]
    return train, test


def get_users(trainset: pd.DataFrame, testset: pd.DataFrame) -> Tuple[set, set]:
    """Return a list of users in both train and test sets, and users in both."""
    train_users = set(trainset["User"].unique())
    test_users = set(testset["User"].unique())
    train_test_users = train_users.intersection(test_users)
    test_only_users = test_users.difference(train_users)
    return train_test_users, test_only_users


def add_train_transacations_to_testset(
        trainset: pd.DataFrame,
        testset: pd.DataFrame,
        train_test_users: set,
        sample_size: int = 10,
        consider_card: bool = False,
) -> pd.DataFrame:
    """
    Add a sampling of transactions from the trainset for each trainset user that also
    appears in the testset.
    """
    groupby_columns = ["User", "Card"] if consider_card else ["User"]
    sort_columns = (
        ["User", "Card", "total_minutes"]
        if consider_card
        else ["User", "total_minutes"]
    )
    # Get the indices of the last y-1 transactions for each user in dataframe x
    get_sample_indices = lambda x, y: x.index[-(y - 1):]

    test_extra_indices = (
        trainset.loc[trainset["User"].isin(train_test_users)]
            .groupby(groupby_columns)
            .apply(get_sample_indices, sample_size)
    )
    test_extra_indices = test_extra_indices.explode()
    testset = pd.concat([trainset.loc[test_extra_indices], testset])
    testset.sort_values(by=sort_columns, inplace=True)
    return testset


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


def prepare_dataset(cfg, data: pd.DataFrame) -> pd.DataFrame:
    """
    Return a preprocessed train-test split of the data.
    """

    data = do_basic_preprocessing(data, cfg.consider_card)

    # log.info("Splitting into train and test sets...")
    # trainset, testset = get_train_test_split(data, cfg.train_test_split_year)

    # Monte note: I don't know why we're doing this. Isn't this bad practice?
    # log.info("Adding train transactions to test set...")
    # train_test_users, test_only_users = get_users(trainset, testset)
    # testset = add_train_transacations_to_testset(
    #     trainset, testset, train_test_users, cfg.sample_size, cfg.consider_card
    # )
    # log.info("Adding static user fields...")
    # trainset, testset = add_static_user_fields(
    #     trainset, testset, test_only_users, cfg.consider_card
    # )

    return data

def save_processed_dataset(dataset: pd.DataFrame, cfg, raw_data_dir_name="data"):
    data_dir = os.path.join(get_original_cwd(), raw_data_dir_name)
    filepath = os.path.join(data_dir, cfg.name)
    os.makedirs(data_dir, exist_ok=True)
    filepath = f"{filepath}.csv"
    dataset.to_csv(f"{filepath}")
    return filepath


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