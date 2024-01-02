"""
Download and preprocess this dataset with:

```
python midi_lm/datasets/nes.py download
python midi_lm/datasets/nes.py split
```
"""


import json
import tarfile
from pathlib import Path

from midi_lm import logger
from midi_lm.datasets import download_google_drive_file, verify_download_hash
from midi_lm.datasets.base import MusicDataModule, MusicDataset, process_dataset

GDRIVE_FILE_ID = "1tyDEwe0exW4xU1W7ZzUMWoTv8K43jF_5"
DATASET_MD5 = "3f3e8ab4f660dd1b19350e5a8a91f3e6"


class NESDataset(MusicDataset):
    name = "NES Music Database"
    author = "Chris Donahue, Huanru Henry Mao, and Julian McAuley"
    source = "https://github.com/chrisdonahue/nesmdb"
    extension = ".mid"

    @classmethod
    def download(cls, output_dir: str | Path):
        # check if file exists
        tar_file = Path(output_dir) / "nesmdb_midi.tar.gz"
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

    @classmethod
    def make_splits(
        cls,
        dataset_dir: str | Path,
        preprocess: bool = True,
    ):
        dataset_dir = Path(dataset_dir)
        train_files = list(dataset_dir.rglob(f"nesmdb_midi/train/*{cls.extension}"))
        val_files = list(dataset_dir.rglob(f"nesmdb_midi/valid/*{cls.extension}"))
        original_file_count = len(train_files) + len(val_files)

        if preprocess:
            logger.info("Preprocessing MIDI files...")
            train_files = process_dataset(train_files)
            val_files = process_dataset(val_files)
        processed_file_count = len(train_files) + len(val_files)

        train_files = [f.relative_to(dataset_dir).as_posix() for f in train_files]
        val_files = [f.relative_to(dataset_dir).as_posix() for f in val_files]

        output = {
            "train": train_files,
            "val": val_files,
            "metadata": {
                "file_extension": cls.extension,
                "split": "data was split by the authors of the dataset",
                "n_skipped_files": original_file_count - processed_file_count,
            },
        }
        output_file = Path(dataset_dir) / "splits.json"
        output_file.write_text(json.dumps(output, indent=2))


class NESDataModule(MusicDataModule):
    dataset_class = NESDataset


if __name__ == "__main__":
    import typer

    app = typer.Typer()

    @app.command()
    def download(output_dir: str = "data/nes"):
        print("Downloading dataset...")
        NESDataset.download(output_dir)

    @app.command()
    def split(dataset_dir: str = "data/nes", preprocess: bool = True):
        print("Making train/validation splits based on the available MIDI files...")
        NESDataset.make_splits(dataset_dir=dataset_dir, preprocess=preprocess)

    app()
