import json

import muspy
import pytest
import torch

from midi_lm.datasets.base import MusicDataset
from midi_lm.tokenizers.base import BaseTokenizer


@pytest.fixture()
def tokenizer():
    class DummyTokenizer(BaseTokenizer):
        def encode(self, music):
            return {"tokens": torch.tensor([1, 2, 3])}

        def decode(self, tokens):
            return muspy.Music()

        def save(self, filepath):
            pass

        @classmethod
        def load(cls, filepath):
            pass

    return DummyTokenizer()


@pytest.fixture()
def dataset_dir(tmp_path):
    return tmp_path


@pytest.fixture()
def midi_files(dataset_dir):
    midi_files = [
        dataset_dir / "file1.mid",
        dataset_dir / "file2.mid",
        dataset_dir / "file3.mid",
    ]
    for file in midi_files:
        track = muspy.Track(
            program=0,
            notes=[
                muspy.Note(time=0, pitch=60, duration=12, velocity=64),
                muspy.Note(time=0, pitch=62, duration=8, velocity=64),
                muspy.Note(time=20, pitch=64, duration=12, velocity=64),
                muspy.Note(time=30, pitch=60, duration=4, velocity=64),
            ],
        )
        muspy.Music(tracks=[track], resolution=12).write(file, kind="midi")
    return midi_files


def test_music_dataset_init(dataset_dir, tokenizer, midi_files):
    dataset = MusicDataset(dataset_dir, tokenizer)
    assert dataset.dataset_dir == dataset_dir
    assert dataset.tokenizer == tokenizer


def test_music_dataset_len(dataset_dir, tokenizer, midi_files):
    dataset = MusicDataset(dataset_dir, tokenizer)
    assert len(dataset) == len(midi_files)


def test_music_dataset_getitem(dataset_dir, tokenizer, midi_files):
    dataset = MusicDataset(dataset_dir, tokenizer)
    for i in range(len(midi_files)):
        item = dataset[i]
        assert isinstance(item, dict)
        assert "_filename" in item
        assert item["_filename"] in midi_files


def test_music_dataset_collect_midi_files(dataset_dir, tokenizer, midi_files):
    # Test the collect_midi_files method of MusicDataset
    dataset = MusicDataset(dataset_dir, tokenizer)
    collected_files = dataset.collect_midi_files(dataset_dir)
    assert set(collected_files) == set(midi_files)


def test_music_dataset_download(dataset_dir, tokenizer):
    # Test the download method of MusicDataset
    with pytest.raises(NotImplementedError):
        MusicDataset.download(dataset_dir)


def test_music_dataset_make_splits(dataset_dir, midi_files):
    MusicDataset.make_splits(dataset_dir, train_split=0.66, seed=0)
    assert (dataset_dir / "splits.json").exists()
    split_file = json.loads((dataset_dir / "splits.json").read_text())
    print(split_file)

    assert len(split_file["train"]) == 2
    assert len(split_file["val"]) == 1
    assert len(split_file["train"]) + len(split_file["val"]) == len(midi_files)
    assert split_file["metadata"]["file_extension"] == ".mid"
    assert split_file["metadata"]["train_split"] == 0.66
    assert split_file["metadata"]["seed"] == 0
