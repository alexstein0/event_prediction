import os
import tarfile

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


def get_repo_root(curr_dir=None):
    if curr_dir is None:
        curr_dir = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(curr_dir)
    # Look for the existence of a .git directory in the current directory and return it if found
    if os.path.isdir(os.path.join(curr_dir, ".git")):
        return curr_dir
    elif curr_dir == parent:
        # We are at the root of the file system and did not find a .git directory. Just return an empty string and it will be fine.
        return ""
    # Recursively call the function on the next level up
    return get_repo_root(parent)


def download(url, download_dir="data"):
    filename = os.path.basename(url)
    os.makedirs(download_dir, exist_ok=True)
    filepath = os.path.join(download_dir, filename)
    if not os.path.exists(filepath):
        try:
            response = requests.get(url, stream=True)
            file_size = int(response.headers.get("Content-Length", 0))
            # Initialize a downloader with a progress bar
            downloader = tqdm(
                response.iter_content(1024),
                f"Downloading {url}",
                total=file_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            )
            with open(filepath, "wb") as file:
                for data in downloader.iterable:
                    # Write data read to the file
                    file.write(data)
                    # Update the progress bar manually
                    downloader.update(len(data))
        except Exception as e:
            print(f"Error: {e}")
    return filepath


def extract(filepath):
    print(f"Checking {filepath} to extract files...")
    dir = os.path.dirname(filepath)
    extracted_files = []
    with tarfile.open(filepath) as tar:
        for member in tar.getmembers():
            if not os.path.isfile(os.path.join(dir, member.name)):
                tar.extract(member, path=dir)
            extracted_files.append(os.path.join(dir, member.name))
    print(f"Extracted files: {extracted_files}")
    return extracted_files


def get_timestamps(X):
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


def add_hours_total_minutes(X):
    """Return a dataframe with new columns 'Hour' and 'total_minutes'"""
    timestamps = get_timestamps(X)
    X["Hour"] = timestamps.dt.hour
    # Add a column for total minutes from timestamp=0 to our datafrae
    zero_time = pd.to_datetime(np.zeros(len(X)))
    total_seconds = (timestamps - zero_time).dt.total_seconds().astype(int)
    total_minutes = total_seconds // 60
    X["total_minutes"] = total_minutes
    return X


def convert_dollars_to_floats(X, log_scale=True):
    """Return a dataframe with the 'Amount' column converted to floats"""
    X["Amount"] = X["Amount"].str.replace("$", "").astype(float)
    if log_scale:
        X["Amount"] = np.log(X["Amount"])
    return X


def do_basic_preprocessing(X, consider_card=False):
    """Return a preprocessed dataframe"""
    X = add_hours_total_minutes(X)
    sort_columns = (
        ["User", "Card", "total_minutes"]
        if consider_card
        else ["User", "total_minutes"]
    )
    X = X.sort_values(by=sort_columns)
    # Add a column numbering the transactions in order
    X["rownumber"] = np.arange(len(X))
    X = convert_dollars_to_floats(X, log_scale=True)
    return X


def get_train_test_split(X, split_year=2018):
    """Return a train-test split of the data based on a single year cutoff"""
    train = X.loc[X["Year"] < split_year]
    test = X.loc[X["Year"] >= split_year]
    return train, test


def get_users(trainset, testset):
    """Return a list of users in both train and test sets, and users in both."""
    train_users = set(trainset["User"].unique())
    test_users = set(testset["User"].unique())
    train_test_users = train_users.intersection(test_users)
    test_only_users = test_users.difference(train_users)
    return train_test_users, test_only_users


def add_train_transacations_to_testset(
    trainset, testset, train_test_users, sample_size=10, consider_card=False
):
    """Add a sampling of transactions from the trainset for each trainset user that also appears in the testset."""
    groupby_columns = ["User", "Card"] if consider_card else ["User"]
    sort_columns = (
        ["User", "Card", "total_minutes"]
        if consider_card
        else ["User", "total_minutes"]
    )
    # Get the indices of the last y-1 transactions for each user in dataframe x
    get_sample_indices = lambda x, y: x.index[-(y - 1) :]

    test_extra_indices = (
        trainset.loc[trainset["User"].isin(train_test_users)]
        .groupby(groupby_columns)
        .apply(get_sample_indices, sample_size)
    )
    test_extra_indices = test_extra_indices.explode()
    testset = pd.concat([trainset.loc[test_extra_indices], testset])
    testset.sort_values(by=sort_columns, inplace=True)
    return testset


def add_static_user_fields(trainset, testset, test_only_users, consider_card=False):
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


def get_data(
    url, train_test_split_year=2018, sample_size=10, consider_card=False, save_csv=False
):
    """Return a preprocessed train-test split of the data."""
    data_dir = os.path.join(get_repo_root(), "data")
    tarfile = download(url, data_dir)
    extracted_files = extract(tarfile)
    data = pd.read_csv(extracted_files[0])
    
    print("Doing basic preprocessing...")
    data = do_basic_preprocessing(data)
    
    print("Splitting into train and test sets...")
    trainset, testset = get_train_test_split(data)

    print("Adding train transactions to test set...")
    # Monte note: I don't know why we're doing this. Isn't this bad practice?
    train_test_users, test_only_users = get_users(trainset, testset)
    testset = add_train_transacations_to_testset(
        trainset, testset, train_test_users, sample_size=10
    )

    print("Adding static user fields...")
    trainset, testset = add_static_user_fields(trainset, testset, test_only_users)
    
    if save_csv:
        print("Saving train and test sets to csv...")
        datafile_base, datafile_ext = os.path.splitext(extracted_files[0])
        train_path = datafile_base + "_train" + datafile_ext
        test_path = datafile_base + "_test" + datafile_ext
        trainset.to_csv(train_path, index=False)
        testset.to_csv(test_path, index=False)
    
    return trainset, testset


if __name__ == "__main__":
    url = "https://obj.umiacs.umd.edu/eventprediction/transactions.tgz"
    train_test_split_year = 2018
    sample_size = 10
    consider_card = False
    _, _ = get_data(
        url, train_test_split_year, sample_size, consider_card, save_csv=True
    )
