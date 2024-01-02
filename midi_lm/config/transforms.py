from dataclasses import dataclass

from midi_lm.transforms.crop import RandomCrop
from midi_lm.transforms.transpose import TransposeNotes


@dataclass
class TransformsConfig:
    crop_n_beats: int | None = None
    transpose_min_semitones: int | None = None
    transpose_max_semitones: int | None = None


@dataclass
class CropConfig(TransformsConfig):
    crop_n_beats: int = 16


@dataclass
class CropTransposeConfig(TransformsConfig):
    crop_n_beats: int = 6 * 4
    transpose_min_semitones: int = -5
    transpose_max_semitones: int = 5


def create_transforms(config: TransformsConfig) -> list:
    transforms = []
    if config.crop_n_beats:
        transforms.append(RandomCrop(config.crop_n_beats))
    if config.transpose_min_semitones and config.transpose_max_semitones:
        transforms.append(
            TransposeNotes(
                min_semitones=config.transpose_min_semitones, max_semitones=config.transpose_max_semitones
            )
        )
    return transforms
