import muspy
import pytest
import torch

from midi_lm.tokenizers.tpd import TimeShiftPitchDurationTokenizer

TEST_KNOWN_TIME_SHIFTS = [0, 10, 20, 30]
TEST_KNOWN_DURATIONS = [2, 4, 8, 12, 16, 32]


@pytest.fixture()
def tokenizer() -> TimeShiftPitchDurationTokenizer:
    return TimeShiftPitchDurationTokenizer(
        known_durations=TEST_KNOWN_DURATIONS, known_time_shifts=TEST_KNOWN_TIME_SHIFTS
    )


@pytest.fixture()
def example_music() -> muspy.Music:
    metadata = muspy.Metadata()
    tempo = muspy.Tempo(time=0, qpm=100)
    piano_track = muspy.Track(
        program=0,
        notes=[
            muspy.Note(time=0, pitch=60, duration=12, velocity=64),
            muspy.Note(time=0, pitch=62, duration=8, velocity=64),
            muspy.Note(time=20, pitch=64, duration=12, velocity=64),
            muspy.Note(time=30, pitch=60, duration=4, velocity=64),
        ],
    )
    music = muspy.Music(metadata=metadata, resolution=12, tempos=[tempo], tracks=[piano_track])
    return music


@pytest.fixture()
def example_tokens() -> dict:
    codes = torch.tensor(
        [
            # bos
            [0, 1, 0],
            # time shift, pitch, duration
            # time shift and durations are offset by 1 for padding
            # pitch is offset by 3 for bos, eos, and padding
            [TEST_KNOWN_TIME_SHIFTS.index(0) + 1, 60 + 3, TEST_KNOWN_DURATIONS.index(12) + 1],
            [TEST_KNOWN_TIME_SHIFTS.index(0) + 1, 62 + 3, TEST_KNOWN_DURATIONS.index(8) + 1],
            [TEST_KNOWN_TIME_SHIFTS.index(20) + 1, 64 + 3, TEST_KNOWN_DURATIONS.index(12) + 1],
            [TEST_KNOWN_TIME_SHIFTS.index(10) + 1, 60 + 3, TEST_KNOWN_DURATIONS.index(4) + 1],
            # eos
            [0, 2, 0],
        ],
        dtype=torch.long,
    )
    return {
        "time_shift": codes[:, 0],
        "pitch": codes[:, 1],
        "duration": codes[:, 2],
    }


def test_encode(
    tokenizer: TimeShiftPitchDurationTokenizer,
    example_music: muspy.Music,
    example_tokens: dict[str, torch.Tensor],
):
    tokens = tokenizer.encode(example_music)
    for key in tokens:
        assert torch.all(tokens[key] == example_tokens[key])


def test_decode(
    tokenizer: TimeShiftPitchDurationTokenizer,
    example_music: muspy.Music,
    example_tokens: dict[str, torch.Tensor],
):
    music = tokenizer.decode(example_tokens)
    assert music == example_music


def test_encode_decode(
    tokenizer: TimeShiftPitchDurationTokenizer,
    example_music: muspy.Music,
):
    # NOTE: this test only works for music where all note durations are in the list of known durations
    assert tokenizer.decode(tokenizer.encode(example_music)) == example_music


def test_save_load(tmp_path, tokenizer: TimeShiftPitchDurationTokenizer):
    output_path = tmp_path / "tokenizer.json"
    tokenizer.save(output_path)
    loaded_tokenizer = TimeShiftPitchDurationTokenizer.load(output_path)
    assert loaded_tokenizer == tokenizer


def test_vocab_size(tokenizer: TimeShiftPitchDurationTokenizer):
    assert tokenizer.vocab_size == {
        "time_shift": 4 + 1,
        "pitch": 128 + 3,
        "duration": 6 + 1,
    }


def test_truncate_sequence(example_music: muspy.Music):
    tokenizer = TimeShiftPitchDurationTokenizer(
        known_durations=TEST_KNOWN_DURATIONS, known_time_shifts=TEST_KNOWN_TIME_SHIFTS, max_seq_len=4
    )
    tokens = tokenizer.encode(example_music)

    expected_codes = torch.tensor(
        [
            # bos
            [0, 1, 0],
            # time shift, pitch, duration
            # time shift and durations are offset by 1 for padding
            # pitch is offset by 3 for bos, eos, and padding
            [TEST_KNOWN_TIME_SHIFTS.index(0) + 1, 60 + 3, TEST_KNOWN_DURATIONS.index(12) + 1],
            [TEST_KNOWN_TIME_SHIFTS.index(0) + 1, 62 + 3, TEST_KNOWN_DURATIONS.index(8) + 1],
            # eos
            [0, 2, 0],
        ],
        dtype=torch.long,
    )
    expected_results = {
        "time_shift": expected_codes[:, 0],
        "pitch": expected_codes[:, 1],
        "duration": expected_codes[:, 2],
    }
    for key in tokens:
        assert torch.all(tokens[key] == expected_results[key])
