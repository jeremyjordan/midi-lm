"""
Download and preprocess this dataset with:

```
python midi_lm/datasets/scales.py download
python midi_lm/datasets/scales.py split
```
"""

import tarfile
from pathlib import Path

from midi_lm import logger
from midi_lm.datasets import download_google_drive_file, verify_download_hash
from midi_lm.datasets.base import MusicDataModule, MusicDataset

GDRIVE_FILE_ID = "1EfWAm_OiHTQUo75id01jQQv-TWd02-MU"
DATASET_MD5 = "3a59cdfa08ac1a4b2f156cbc88c6ce6e"


class ScalesDataset(MusicDataset):
    name = "Major and Minor Scales"
    author = "Jeremy Jordan"
    source = "Created in Logic Pro"
    extension = ".mid"

    @classmethod
    def download(cls, output_dir: str | Path):
        # check if file exists
        tar_file = Path(output_dir) / "scales.tar.gz"
        if tar_file.exists():
            logger.info(f"Found {tar_file}, skipping download...")
        else:
            logger.info(f"Downloading {cls.name} dataset...")
            tar_file.parent.mkdir(parents=True, exist_ok=True)
            download_google_drive_file(GDRIVE_FILE_ID, tar_file.as_posix())
        # check file hash to ensure it's the correct file
        verify_download_hash(tar_file, expected_hash=DATASET_MD5)
        with tarfile.open(tar_file) as tar:
            logger.info(f"Extracting {tar_file.parent}...")
            tar.extractall(tar_file.parent)


class ScalesDataModule(MusicDataModule):
    dataset_class = ScalesDataset


if __name__ == "__main__":
    import typer

    app = typer.Typer()

    @app.command()
    def download(output_dir: str = "data/scales"):
        print("Downloading dataset...")
        ScalesDataset.download(output_dir)

    @app.command()
    def split(dataset_dir: str = "data/scales", preprocess: bool = True):
        print("Making train/validation splits based on the available MIDI files...")
        ScalesDataset.make_splits(dataset_dir=dataset_dir, train_split=0.8, preprocess=preprocess)

    app()
