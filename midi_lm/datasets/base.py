import json
import math
import multiprocessing
from pathlib import Path

import muspy
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from midi_lm import logger
from midi_lm.tokenizers.base import BaseTokenizer
from midi_lm.transforms import Compose

__all__ = ["MusicDataset", "MusicDataModule"]


def get_midi(midi_file: str | Path):
    midi_file = Path(midi_file)
    if midi_file.suffix == ".json":
        return muspy.load_json(midi_file)
    return muspy.read_midi(midi_file)


def _is_valid_midi_file(music: muspy.Music):
    sufficent_beats = music.get_end_time() >= 2 * music.resolution
    sufficent_tracks = len(music.tracks) >= 1
    sufficent_instruments = len(set(track.program for track in music.tracks)) >= 1
    sufficent_pitches = muspy.n_pitches_used(music) >= 2
    return sufficent_beats and sufficent_tracks and sufficent_instruments and sufficent_pitches


def _process_midi_file(midi_file) -> Path | None:
    music = get_midi(midi_file)
    if not _is_valid_midi_file(music):
        logger.warning(f"Skipping {midi_file}, not a valid MIDI file...")
        return None

    music = music.adjust_resolution(12)  # TODO: make this resolution configurable eventually
    for track in music.tracks:
        track.sort()
    output_file = midi_file.with_suffix(".json")
    music.save(output_file)
    return output_file


def process_dataset(midi_files: list[Path]) -> list[Path]:
    logger.info(f"Processing {len(midi_files)} MIDI files...")

    with multiprocessing.Pool() as pool:
        output = list(
            tqdm(
                pool.imap_unordered(_process_midi_file, midi_files),
                total=len(midi_files),
            )
        )

    return [f for f in output if f is not None]


class MusicDataset(Dataset):
    name: str = ""
    author: str = ""
    source: str = ""
    extension: str = ".mid"

    def __init__(
        self,
        dataset_dir: str,
        tokenizer: BaseTokenizer,
        transforms: list | None = None,
        split: str | None = None,
        split_file: Path | str | None = None,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.midi_files = self.collect_midi_files(self.dataset_dir, split_file=split_file, split=split)
        assert len(self.midi_files) > 0, f"No MIDI files found in {self.dataset_dir}"
        self.transforms = Compose(transforms) if transforms else None
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.midi_files)

    def __getitem__(self, index):
        filename = self.midi_files[index]
        music = get_midi(filename)
        if self.transforms:
            music = self.transforms(music)
        tokens = self.tokenizer.encode(music=music)
        return {
            **tokens,
            "_filename": filename,
        }

    def collect_midi_files(
        self,
        dataset_dir: Path,
        split: str | None = None,
        split_file: Path | str | None = None,
    ):
        if split_file:
            split_file = Path(dataset_dir, split_file)
            assert split in ["train", "val"], "Split must be either 'train' or 'val'"
            filenames = json.loads(split_file.read_text())[split]
            midi_files = [dataset_dir / filename for filename in filenames]
        else:
            midi_files = list(dataset_dir.glob(f"**/*{self.extension}"))
        return midi_files

    @classmethod
    def download(cls, output_dir: str | Path):
        raise NotImplementedError("This dataset is not available for download.")

    @classmethod
    def make_splits(
        cls,
        dataset_dir: str | Path,
        train_split: float = 0.8,
        seed: int = 1337,
        preprocess: bool = True,
    ):
        dataset_dir = Path(dataset_dir)
        midi_files = list(dataset_dir.rglob(f"*{cls.extension}"))
        logger.info(f"Found {len(midi_files)} midi files in {dataset_dir}")
        original_file_count = len(midi_files)

        if preprocess:
            logger.info("Preprocessing MIDI files...")
            midi_files = process_dataset(midi_files)
        processed_file_count = len(midi_files)

        # split into train and val
        gen = torch.Generator()
        gen.manual_seed(seed)
        indices = torch.randperm(len(midi_files), generator=gen).tolist()
        split = math.ceil(len(indices) * train_split)  # round up to nearest integer
        train_indices = indices[:split]
        val_indices = indices[split:]
        train_files = [midi_files[i].relative_to(dataset_dir).as_posix() for i in train_indices]
        val_files = [midi_files[i].relative_to(dataset_dir).as_posix() for i in val_indices]

        output = {
            "train": train_files,
            "val": val_files,
            "metadata": {
                "file_extension": cls.extension,
                "train_split": train_split,
                "seed": seed,
                "n_skipped_files": original_file_count - processed_file_count,
            },
        }
        output_file = Path(dataset_dir) / "splits.json"
        output_file.write_text(json.dumps(output, indent=2))


class MusicDataModule(LightningDataModule):
    dataset_class: type[MusicDataset]

    def __init__(
        self,
        dataset_dir,
        tokenizer,
        transforms=None,
        batch_size=32,
        num_workers=8,
        collate_fn=None,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.tokenizer = tokenizer
        self.transforms = transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn

    def setup(self, stage=None):
        self.train_dataset = self.dataset_class(
            self.dataset_dir,
            self.tokenizer,
            transforms=self.transforms,
            split_file="splits.json",
            split="train",
        )
        self.val_dataset = self.dataset_class(
            self.dataset_dir,
            self.tokenizer,
            transforms=self.transforms,
            split_file="splits.json",
            split="val",
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            drop_last=True,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            drop_last=False,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
