import muspy
import pytest

from midi_lm.transforms.crop import RandomCrop, crop_music


def assert_music_equal(first: muspy.Music, second: muspy.Music):
    assert len(first.tracks) == len(second.tracks)
    for first_track, second_track in zip(first.tracks, second.tracks, strict=True):
        assert len(first_track.notes) == len(second_track.notes)
        for first_note, second_note in zip(first_track.notes, second_track.notes, strict=True):
            assert first_note == second_note


@pytest.mark.parametrize(
    ("start_beat", "end_beat", "expected_music"),
    [
        (
            0,  # start_beat
            10,  # end_beat
            muspy.Music(
                resolution=2,
                tracks=[
                    muspy.Track(
                        program=0,
                        name="A major scale",
                        notes=[
                            muspy.Note(time=0, pitch=69, duration=2, velocity=80),
                            muspy.Note(time=2, pitch=71, duration=1, velocity=80),
                            muspy.Note(time=3, pitch=73, duration=1, velocity=80),
                            muspy.Note(time=4, pitch=74, duration=1, velocity=80),
                            muspy.Note(time=5, pitch=76, duration=1, velocity=80),
                            muspy.Note(time=6, pitch=78, duration=1, velocity=80),
                            muspy.Note(time=7, pitch=80, duration=1, velocity=80),
                            muspy.Note(time=8, pitch=81, duration=2, velocity=80),
                            muspy.Note(time=10, pitch=80, duration=1, velocity=80),
                            muspy.Note(time=11, pitch=78, duration=1, velocity=80),
                            muspy.Note(time=12, pitch=76, duration=1, velocity=80),
                            muspy.Note(time=13, pitch=74, duration=1, velocity=80),
                            muspy.Note(time=14, pitch=73, duration=1, velocity=80),
                            muspy.Note(time=15, pitch=71, duration=1, velocity=80),
                            muspy.Note(time=16, pitch=69, duration=2, velocity=80),
                        ],
                    )
                ],
            ),
        ),
        (
            3,  # start_beat
            7,  # end_beat
            muspy.Music(
                resolution=2,
                tracks=[
                    muspy.Track(
                        program=0,
                        name="A major scale",
                        notes=[
                            muspy.Note(time=6 - 6, pitch=78, duration=1, velocity=80),
                            muspy.Note(time=7 - 6, pitch=80, duration=1, velocity=80),
                            muspy.Note(time=8 - 6, pitch=81, duration=2, velocity=80),
                            muspy.Note(time=10 - 6, pitch=80, duration=1, velocity=80),
                            muspy.Note(time=11 - 6, pitch=78, duration=1, velocity=80),
                            muspy.Note(time=12 - 6, pitch=76, duration=1, velocity=80),
                            muspy.Note(time=13 - 6, pitch=74, duration=1, velocity=80),
                        ],
                    )
                ],
            ),
        ),
    ],
)
def test_crop_music(
    example_piano_track: muspy.Music,
    start_beat: int,
    end_beat: int,
    expected_music: muspy.Music,
):
    result = crop_music(music=example_piano_track, start_beat=start_beat, end_beat=end_beat)
    assert_music_equal(result, expected_music)


@pytest.mark.parametrize(
    ("start_beat", "end_beat"),
    [
        (0, 0),
        (10, 0),
        (10, 10),
        (-2, 2),
        (2, -2),
    ],
)
def test_crop_music_invalid_input(
    example_piano_track: muspy.Music,
    start_beat: int,
    end_beat: int,
):
    with pytest.raises(AssertionError):
        crop_music(music=example_piano_track, start_beat=start_beat, end_beat=end_beat)


def test_crop_music_does_not_modify_original_music(example_piano_track: muspy.Music):
    original = example_piano_track.deepcopy()
    crop_music(music=example_piano_track, start_beat=0, end_beat=10)
    crop_music(music=example_piano_track, start_beat=3, end_beat=13)
    assert_music_equal(original, example_piano_track)


def test_crop_transform(example_piano_track: muspy.Music):
    original = example_piano_track.deepcopy()
    crop = RandomCrop(n_beats=10)
    result = crop(example_piano_track)
    assert_music_equal(original, example_piano_track)

    assert len(result.tracks[0]) < len(original.tracks[0])
    assert result.get_end_time() <= 10 * result.resolution


def test_crop_short_music():
    piano_track = muspy.Track(
        program=0,
        name="A major scale snippet",
        notes=[
            muspy.Note(time=0, pitch=69, duration=2, velocity=80),
            muspy.Note(time=2, pitch=71, duration=1, velocity=80),
            muspy.Note(time=3, pitch=73, duration=1, velocity=80),
            muspy.Note(time=4, pitch=74, duration=1, velocity=80),
        ],
    )
    music = muspy.Music(resolution=2, tracks=[piano_track])
    crop = RandomCrop(n_beats=20)
    result = crop(music)
    assert result == music
