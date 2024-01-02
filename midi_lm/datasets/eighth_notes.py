"""
Download and preprocess this dataset with:

```
python midi_lm/datasets/eighth_notes.py download
python midi_lm/datasets/eighth_notes.py split
```
"""
from pathlib import Path

import muspy

from midi_lm import logger
from midi_lm.datasets.base import MusicDataModule, MusicDataset


def create_eighth_notes(pitch: int = 64, resolution: int = 12):
    notes = []
    eighth_note_duration = resolution // 2
    for time in range(0, 4 * resolution, eighth_note_duration):
        notes.append(muspy.Note(time=time, pitch=pitch, duration=eighth_note_duration, velocity=64))
    track = muspy.Track(
        program=0,
        notes=notes,
    )
    music = muspy.Music(tracks=[track], resolution=resolution)
    return music


class EighthNotesDataset(MusicDataset):
    name = "Eighth Notes"
    author = "Jeremy Jordan"
    extension = ".mid"

    @classmethod
    def download(cls, output_dir: str | Path):
        logger.info("Creating dataset...")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for pitch in range(60, 72):
            music = create_eighth_notes(pitch=pitch)
            music.write(output_dir / f"{pitch}.mid", kind="midi")


class EighthNotesDataModule(MusicDataModule):
    dataset_class = EighthNotesDataset


if __name__ == "__main__":
    import typer

    app = typer.Typer()

    @app.command()
    def download(output_dir: str = "data/eighth_notes"):
        print("Downloading dataset...")
        EighthNotesDataset.download(output_dir)

    @app.command()
    def split(dataset_dir: str = "data/eighth_notes"):
        print("Making train/validation splits based on the available MIDI files...")
        # we skip preprocessing here because the data is already in a good format
        EighthNotesDataset.make_splits(dataset_dir=dataset_dir, train_split=0.5, preprocess=False)

    app()
