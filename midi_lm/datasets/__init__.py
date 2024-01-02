import hashlib
import zipfile
from pathlib import Path

import requests

from midi_lm import logger


def download_google_drive_file(id: str, destination: str | Path):
    try:
        import gdown

        gdown.download(id=id, output=destination, quiet=False, fuzzy=True)
    except ImportError as e:
        raise ImportError("Please install gdown with `pip install gdown` to download this dataset.") from e


def download_internet_file(url: str, destination: str | Path):
    response = requests.get(url)
    with open(destination, "wb") as f:
        f.write(response.content)


def verify_download_hash(filepath: str | Path, expected_hash: str, hash_algo: str = "md5"):
    hash_obj = hashlib.new(hash_algo)
    with open(filepath, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            hash_obj.update(chunk)
    computed_hash = hash_obj.hexdigest()
    assert computed_hash == expected_hash, f"Hash mismatch: {computed_hash} != {expected_hash}"


def download_and_extract_zipfile(
    url: str,
    output_dir: str | Path,
    filename: str | Path,
    expected_hash: str | None = None,
):
    # check if file exists
    zip_file = Path(output_dir) / filename
    if zip_file.exists():
        logger.info(f"Found {zip_file}, skipping download...")
    else:
        logger.info(f"Downloading dataset to {output_dir}...")
        zip_file.parent.mkdir(parents=True, exist_ok=True)
        download_internet_file(url, zip_file.as_posix())

    if expected_hash:
        # check file hash to ensure it's the correct file
        verify_download_hash(zip_file, expected_hash=expected_hash)

    with zipfile.ZipFile(zip_file) as zip:
        logger.info(f"Extracting {zip_file}...")
        zip.extractall(zip_file.parent)
