import muspy
import pytest

from midi_lm.transforms.transpose import TransposeNotes


@pytest.mark.parametrize(
    "semitone",
    [
        (-3),
        (-2),
        (-1),
        (0),
        (1),
        (2),
        (3),
        (4),
    ],
)
def test_transpose_specific_semitone(example_piano_track: muspy.Music, semitone):
    transform = TransposeNotes(min_semitones=semitone, max_semitones=semitone)
    new_music = transform(example_piano_track.deepcopy())

    for original_track, transposed_track in zip(example_piano_track.tracks, new_music.tracks, strict=True):
        if original_track.is_drum:
            continue
        for original_note, transposed_note in zip(
            original_track.notes, transposed_track.notes, strict=True
        ):
            assert transposed_note.pitch - original_note.pitch == semitone


@pytest.mark.parametrize(
    ("min_semitones", "max_semitones"),
    [
        (1, 2),
        (1, 4),
        (-1, 4),
        (-3, -1),
    ],
)
def test_transpose_random_semitone(example_piano_track: muspy.Music, min_semitones, max_semitones):
    transform = TransposeNotes(min_semitones=min_semitones, max_semitones=max_semitones)
    new_music = transform(example_piano_track.deepcopy())

    shifts = set()
    for original_track, transposed_track in zip(example_piano_track.tracks, new_music.tracks, strict=True):
        if original_track.is_drum:
            continue
        for original_note, transposed_note in zip(
            original_track.notes, transposed_track.notes, strict=True
        ):
            shift = transposed_note.pitch - original_note.pitch
            shifts.add(shift)
    assert len(shifts) == 1, "all notes should be transposed by the same amount"

    # check that the shift is within the specified range
    value = shifts.pop()
    assert min_semitones <= value and value <= max_semitones
