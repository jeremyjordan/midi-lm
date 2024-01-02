"""
Download and preprocess this dataset with:

```
python midi_lm/datasets/symphony_net.py download
python midi_lm/datasets/symphony_net.py split
```
"""

import tarfile
from pathlib import Path

from midi_lm import logger
from midi_lm.datasets import download_google_drive_file, verify_download_hash
from midi_lm.datasets.base import MusicDataModule, MusicDataset

GDRIVE_FILE_ID = "1j9Pvtzaq8k_QIPs8e2ikvCR-BusPluTb"
DATASET_MD5 = "todo"


class SymphonyNetDataset(MusicDataset):
    name = "Symphony Net"
    author = "Jiafeng Liu, Yuanliang Dong, Zehua Cheng, Xinran Zhang, Xiaobing Li, Feng Yu, and Maosong Sun"
    source = "https://symphonynet.github.io/"
    extension = ".mid"

    @classmethod
    def download(cls, output_dir: str | Path):
        # check if file exists
        tar_file = Path(output_dir) / "symphonynet_dataset.tar.gz"
        if tar_file.exists():
            logger.info(f"Found {tar_file}, skipping download...")
            # check file hash to ensure it's the correct file
            verify_download_hash(tar_file, expected_hash=DATASET_MD5)
        else:
            logger.info(f"Downloading {cls.name} dataset...")
            tar_file.parent.mkdir(parents=True, exist_ok=True)
            download_google_drive_file(GDRIVE_FILE_ID, tar_file.as_posix())
        with tarfile.open(tar_file) as tar:
            logger.info(f"Extracting {tar_file.parent}...")
            tar.extractall(tar_file.parent)


class SymphonyNetDataModule(MusicDataModule):
    dataset_class = SymphonyNetDataset


if __name__ == "__main__":
    import typer

    app = typer.Typer()

    @app.command()
    def download(output_dir: str = "data/symphony_net"):
        print("Downloading dataset...")
        SymphonyNetDataset.download(output_dir)

    @app.command()
    def split(dataset_dir: str = "data/symphony_net", preprocess: bool = True):
        print("Making train/validation splits based on the available MIDI files...")
        SymphonyNetDataset.make_splits(dataset_dir=dataset_dir, preprocess=preprocess)

    app()
