"""
Download and preprocess this dataset with:

```
python midi_lm/datasets/bach_chorales.py download
python midi_lm/datasets/bach_chorales.py split
```
"""

import json
from pathlib import Path

from midi_lm import logger
from midi_lm.datasets import download_and_extract_zipfile
from midi_lm.datasets.base import MusicDataModule, MusicDataset, process_dataset

DATASET_MD5 = "2fb72faf2659e82e9de08b16f2cca1e9"
DATASET_URL = "https://web.archive.org/web/20220401114442/http://www-ens.iro.umontreal.ca/~boulanni/JSB%20Chorales.zip"


class BachChoralesDataset(MusicDataset):
    name = "JSB Chorales"
    author = "Nicolas Boulanger-Lewandowski, Yoshua Bengio, and Pascal Vincent"
    source = "https://web.archive.org/web/20220401114442/http://www-ens.iro.umontreal.ca/~boulanni/icml2012"
    extension = ".mid"

    @classmethod
    def download(cls, output_dir: str | Path):
        download_and_extract_zipfile(
            url=DATASET_URL,
            output_dir=output_dir,
            filename="JSB Chorales.zip",
            expected_hash=DATASET_MD5,
        )

    @classmethod
    def make_splits(
        cls,
        dataset_dir: str | Path,
        preprocess: bool = True,
    ):
        dataset_dir = Path(dataset_dir)
        train_files = list(dataset_dir.rglob(f"JSB Chorales/train/*{cls.extension}"))
        val_files = list(dataset_dir.rglob(f"JSB Chorales/valid/*{cls.extension}"))
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


class BachChoralesDataModule(MusicDataModule):
    dataset_class = BachChoralesDataset


if __name__ == "__main__":
    import typer

    app = typer.Typer()

    @app.command()
    def download(output_dir: str = "data/bach_chorales"):
        print("Downloading dataset...")
        BachChoralesDataset.download(output_dir)

    @app.command()
    def split(dataset_dir: str = "data/bach_chorales", preprocess: bool = True):
        print("Making train/validation splits based on the available MIDI files...")
        BachChoralesDataset.make_splits(dataset_dir=dataset_dir, preprocess=preprocess)

    app()
