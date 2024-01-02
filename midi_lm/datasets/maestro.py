"""
Download and preprocess this dataset with:

```
python midi_lm/datasets/maestro.py download
python midi_lm/datasets/maestro.py split
```
"""

import csv
import json
from pathlib import Path

from midi_lm import logger
from midi_lm.datasets import download_and_extract_zipfile, download_internet_file
from midi_lm.datasets.base import MusicDataModule, MusicDataset, process_dataset

DATASET_MD5 = "b7656589d0ff8f1170d13f69837390ba"
DATASET_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
SPLITS_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.csv"


class MaestroDataset(MusicDataset):
    name = "MAESTRO (MIDI and Audio Edited for Synchronous TRacks and Organization)"
    author = "Google Magenta"
    source = "https://magenta.tensorflow.org/datasets/maestro"
    extension = ".midi"

    @classmethod
    def download(cls, output_dir: str | Path):
        download_and_extract_zipfile(
            url=DATASET_URL,
            output_dir=output_dir,
            filename="maestro-v3.0.0-midi.zip",
            expected_hash=DATASET_MD5,
        )

    @classmethod
    def make_splits(
        cls,
        dataset_dir: str | Path,
        preprocess: bool = True,
    ):
        dataset_dir = Path(dataset_dir)
        splits_file = Path(dataset_dir) / "maestro-v3.0.0" / "maestro-v3.0.0.csv"

        # download splits file
        if not splits_file.exists():
            logger.info(f"Downloading {cls.name} splits...")
            splits_file.parent.mkdir(parents=True, exist_ok=True)
            download_internet_file(SPLITS_URL, splits_file.as_posix())

        # read splits file
        splits = {"train": [], "validation": [], "test": []}
        with open(splits_file, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                splits[row["split"]].append(row["midi_filename"])

        train_files = [dataset_dir / "maestro-v3.0.0" / f for f in splits["train"]]
        val_files = [dataset_dir / "maestro-v3.0.0" / f for f in splits["validation"]]
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


class MaestroDataModule(MusicDataModule):
    dataset_class = MaestroDataset


if __name__ == "__main__":
    import typer

    app = typer.Typer()

    @app.command()
    def download(output_dir: str = "data/maestro"):
        print("Downloading dataset...")
        MaestroDataset.download(output_dir)

    @app.command()
    def split(dataset_dir: str = "data/maestro", preprocess: bool = True):
        print("Making train/validation splits based on the available MIDI files...")
        MaestroDataset.make_splits(dataset_dir=dataset_dir, preprocess=preprocess)

    app()
