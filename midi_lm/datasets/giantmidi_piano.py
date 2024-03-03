"""
Download and preprocess this dataset with:

```
python midi_lm/datasets/giantmidi_piano.py download
python midi_lm/datasets/giantmidi_piano.py split
```
"""

import zipfile
from pathlib import Path

from midi_lm import logger
from midi_lm.datasets import download_google_drive_file, verify_download_hash
from midi_lm.datasets.base import MusicDataModule, MusicDataset

GDRIVE_FILE_ID = "1BDEPaEWFEB2ADquS1VYp5iLZYVngw799"
DATASET_MD5 = "aaece5750b0cfe30b6a3be5c7bb14f83"


class GiantMidiPianoDataset(MusicDataset):
    name = "GiantMIDI Piano"
    author = "Qiuqiang Kong, Bochen Li, Jitong Chen, and Yuxuan Wang"
    source = "https://github.com/bytedance/GiantMIDI-Piano"
    extension = ".mid"

    @classmethod
    def download(cls, output_dir: str | Path):
        # check if file exists
        zip_file = Path(output_dir) / "midis_v1.2.zip"
        if zip_file.exists():
            logger.info(f"Found {zip_file}, skipping download...")
            # check file hash to ensure it's the correct file
            verify_download_hash(zip_file, expected_hash=DATASET_MD5)
        else:
            logger.info(f"Downloading {cls.name} dataset...")
            zip_file.parent.mkdir(parents=True, exist_ok=True)
            download_google_drive_file(GDRIVE_FILE_ID, zip_file.as_posix())
        with zipfile.ZipFile(zip_file) as zip:
            logger.info(f"Extracting {zip_file}...")
            zip.extractall(zip_file.parent)


class GiantMidiPianoDataModule(MusicDataModule):
    dataset_class = GiantMidiPianoDataset


if __name__ == "__main__":
    import typer

    app = typer.Typer()

    @app.command()
    def download(output_dir: str = "data/giantmidi_piano"):
        print("Downloading dataset...")
        GiantMidiPianoDataset.download(output_dir)

    @app.command()
    def split(dataset_dir: str = "data/giantmidi_piano", preprocess: bool = True):
        print("Making train/validation splits based on the available MIDI files...")
        GiantMidiPianoDataset.make_splits(dataset_dir=dataset_dir, preprocess=preprocess)

    app()
