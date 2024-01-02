from midi_lm.config.transforms import TransformsConfig, create_transforms
from midi_lm.transforms.crop import RandomCrop
from midi_lm.transforms.transpose import TransposeNotes


def test_create_transforms_crop_transpose():
    config = TransformsConfig(crop_n_beats=10, transpose_min_semitones=-5, transpose_max_semitones=5)
    transforms = create_transforms(config)

    assert transforms == [
        RandomCrop(n_beats=10),
        TransposeNotes(min_semitones=-5, max_semitones=5),
    ]


def test_create_transforms_crop():
    config = TransformsConfig(crop_n_beats=10)
    transforms = create_transforms(config)

    assert transforms == [
        RandomCrop(n_beats=10),
    ]


def test_create_transforms_transpose():
    config = TransformsConfig(transpose_min_semitones=-5, transpose_max_semitones=5)
    transforms = create_transforms(config)

    assert transforms == [
        TransposeNotes(min_semitones=-5, max_semitones=5),
    ]


def test_create_transforms_none():
    config = TransformsConfig()
    transforms = create_transforms(config)

    assert transforms == []
