import muspy
import pytest
import torch

from midi_lm.tokenizers.structured_sequence import StructuredSequenceTokenizer


@pytest.fixture()
def tokenizer() -> StructuredSequenceTokenizer:
    return StructuredSequenceTokenizer()


@pytest.fixture()
def example_music() -> muspy.Music:
    metadata = muspy.Metadata()
    tempo = muspy.Tempo(time=0, qpm=100)
    piano_track = muspy.Track(
        program=0,
        notes=[
            muspy.Note(time=0, pitch=60, duration=12, velocity=64),
            muspy.Note(time=0, pitch=62, duration=8, velocity=54),
            muspy.Note(time=20, pitch=64, duration=12, velocity=60),
            muspy.Note(time=30, pitch=60, duration=4, velocity=64),
        ],
    )
    music = muspy.Music(metadata=metadata, resolution=12, tempos=[tempo], tracks=[piano_track])
    return music


@pytest.fixture()
def example_tokens(tokenizer: StructuredSequenceTokenizer) -> dict:
    # velocities are quantized to 32 bins
    codes = torch.tensor(
        [
            # BOS
            tokenizer.config.bos_token_id,
            # pitch, velocity, duration, time_shift
            # note 1
            60 + tokenizer.pitch_offset,
            16 + tokenizer.velocity_offset,  # idx 16 -> 66, closest to 64
            (12 - 1) + tokenizer.duration_offset,  # duration starts at 1
            0 + tokenizer.time_shift_offset,
            # note 2
            62 + tokenizer.pitch_offset,
            13 + tokenizer.velocity_offset,  # idx 13 -> 53, closest to 54
            (8 - 1) + tokenizer.duration_offset,  # duration starts at 1
            0 + tokenizer.time_shift_offset,
            # note 3
            64 + tokenizer.pitch_offset,
            15 + tokenizer.velocity_offset,  # idx 15 -> 61, closest to 60
            (12 - 1) + tokenizer.duration_offset,  # duration starts at 1
            20 + tokenizer.time_shift_offset,
            # note 4
            60 + tokenizer.pitch_offset,
            16 + tokenizer.velocity_offset,  # idx 16 -> 66, closest to 64
            (4 - 1) + tokenizer.duration_offset,  # duration starts at 1
            10 + tokenizer.time_shift_offset,
            # EOS
            tokenizer.config.eos_token_id,
        ],
        dtype=torch.long,
    )
    return {
        "token_ids": codes,
    }


def test_encode(
    tokenizer: StructuredSequenceTokenizer,
    example_music: muspy.Music,
    example_tokens: dict[str, torch.Tensor],
):
    tokens = tokenizer.encode(example_music)
    for key in tokens:
        assert torch.all(tokens[key] == example_tokens[key])


def test_decode(
    tokenizer: StructuredSequenceTokenizer,
    example_music: muspy.Music,
    example_tokens: dict[str, torch.Tensor],
):
    music = tokenizer.decode(example_tokens)
    for note1, note2 in zip(music.tracks[0].notes, example_music.tracks[0].notes, strict=True):
        assert note1.pitch == note2.pitch
        assert note1.time == note2.time
        assert note1.duration == note2.duration
        # binned velocity is not exact
        assert abs(note1.velocity - note2.velocity) < 3


def test_token_types(tokenizer: StructuredSequenceTokenizer):
    result = tokenizer.token_types

    # assert that index ranges are contiguous
    assert result["special"][1] + 1 == result["pitch"][0]
    assert result["pitch"][1] + 1 == result["velocity"][0]
    assert result["velocity"][1] + 1 == result["duration"][0]
    assert result["duration"][1] + 1 == result["timeshift"][0]
    assert result["timeshift"][1] + 1 == tokenizer.vocab_size

    # assert that index ranges are correct
    assert result["special"][1] - result["special"][0] == 3 - 1
    assert result["pitch"][1] - result["pitch"][0] == 128 - 1
    assert result["velocity"][1] - result["velocity"][0] == 32 - 1
    assert result["duration"][1] - result["duration"][0] == 256 - 1
    assert result["timeshift"][1] - result["timeshift"][0] == 257 - 1


def test_boundary_tokens(tokenizer: StructuredSequenceTokenizer):
    metadata = muspy.Metadata()
    tempo = muspy.Tempo(time=0, qpm=100)
    piano_track = muspy.Track(
        program=0,
        notes=[
            muspy.Note(time=0, pitch=0, duration=1, velocity=127),
            muspy.Note(time=0, pitch=127, duration=300, velocity=1),
            muspy.Note(time=256, pitch=64, duration=12, velocity=60),
        ],
    )
    music = muspy.Music(metadata=metadata, resolution=12, tempos=[tempo], tracks=[piano_track])
    tokens = tokenizer.encode(music)
    expected_tokens = torch.tensor(
        [
            # BOS
            tokenizer.config.bos_token_id,
            # pitch, velocity, duration, time_shift
            # note 1
            0 + tokenizer.pitch_offset,
            31 + tokenizer.velocity_offset,  # idx 31 -> 127
            (1 - 1) + tokenizer.duration_offset,  # duration starts at 1
            0 + tokenizer.time_shift_offset,
            # note 2
            127 + tokenizer.pitch_offset,
            0 + tokenizer.velocity_offset,  # idx 0 -> 1
            (256 - 1) + tokenizer.duration_offset,  # duration starts at 1
            0 + tokenizer.time_shift_offset,
            # note 3
            64 + tokenizer.pitch_offset,
            15 + tokenizer.velocity_offset,  # idx 15 -> 61, closest to 60
            (12 - 1) + tokenizer.duration_offset,  # duration starts at 1
            256 + tokenizer.time_shift_offset,
            # EOS
            tokenizer.config.eos_token_id,
        ],
        dtype=torch.long,
    )
    assert torch.all(tokens["token_ids"] == expected_tokens)
