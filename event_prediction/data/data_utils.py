import os
import requests
from tqdm import tqdm
import tarfile
import logging
from torch.utils.data import Dataset
from hydra.utils import get_original_cwd
log = logging.getLogger(__name__)


def get_data(cfg, data_dir_name="data"):
    data_dir = os.path.join(get_original_cwd(), data_dir_name)
    os.makedirs(data_dir, exist_ok=True)

    file_path = os.path.join(data_dir, cfg.name)
    try:
        data = get_data_from_file(file_path)
    except:
        download_and_save_data_from_url(cfg.url, file_path)
        data = get_data_from_file(file_path)
    return data


def download_and_save_data_from_url(url: str, filepath: str) -> str:
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
            log.info(f"Dataset saved to {filepath}")

    except Exception as e:
        log.error(f"Error: {e}")
    return filepath

def get_data_from_file(filepath: str):
    log.info(f"Checking {filepath} to load data...")

    with tarfile.open(filepath, "r:gz") as tar:
        for member in tar.getmembers():
            f = tar.extractfile(member)
            if f is not None:
                content = f.read()
        log.info(f"Data load complete")
        return content

# def get_data_from_path(filepath: str):
#     # todo, is there a way to extract directly from the zip instead of having to save as a csv?
#     log.info(f"Checking {filepath} to extract files...")
#     dir = os.path.dirname(filepath)
#     extracted_files = []
#     with tarfile.open(filepath) as tar:
#         for member in tar.getmembers():
#             if not os.path.isfile(os.path.join(dir, member.name)):
#                 tar.extract(member, path=dir)
#             extracted_files.append(member.name)
#     log.info(f"Extracted files: {extracted_files}")
#     return extracted_files
