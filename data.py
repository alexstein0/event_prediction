import os
import tarfile

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



if __name__ == "__main__":
    url = "https://obj.umiacs.umd.edu/eventprediction/transactions.tgz"
    data_dir = os.path.join(get_repo_root(), "data")
    filepath = download(url, data_dir)
    extract(filepath)
