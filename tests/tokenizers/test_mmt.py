import muspy
import pytest
import torch

from midi_lm.tokenizers.mmt import MultitrackMusicTransformerTokenizer


@pytest.fixture()
def tokenizer() -> MultitrackMusicTransformerTokenizer:
    return MultitrackMusicTransformerTokenizer()


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
    token_offset = 1
    codes = torch.tensor(
        [
            # event_type, beat, position, pitch, duration, instrument
            # start of song
            [0, 0, 0, 0, 0, 0],
            # instruments
            [1, 0, 0, 0, 0, 0 + token_offset],
            # start of notes
            [2, 0, 0, 0, 0, 0],
            # notes
            # duration maps to the closest allowed duration
            [3, 0 + token_offset, 0 + token_offset, 60 + token_offset, 11 + token_offset, 0 + token_offset],
            [3, 0 + token_offset, 0 + token_offset, 62 + token_offset, 7 + token_offset, 0 + token_offset],
            [3, 1 + token_offset, 8 + token_offset, 64 + token_offset, 11 + token_offset, 0 + token_offset],
            [3, 2 + token_offset, 6 + token_offset, 60 + token_offset, 3 + token_offset, 0 + token_offset],
            # end of song
            [4, 0, 0, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    return {
        "event_type": codes[:, 0],
        "beat": codes[:, 1],
        "position": codes[:, 2],
        "pitch": codes[:, 3],
        "duration": codes[:, 4],
        "instrument": codes[:, 5],
    }


def test_encode(
    tokenizer: MultitrackMusicTransformerTokenizer,
    example_music: muspy.Music,
    example_tokens: dict[str, torch.Tensor],
):
    tokens = tokenizer.encode(example_music)
    for key in tokens:
        assert torch.all(tokens[key] == example_tokens[key])


def test_decode(
    tokenizer: MultitrackMusicTransformerTokenizer,
    example_music: muspy.Music,
    example_tokens: dict[str, torch.Tensor],
):
    music = tokenizer.decode(example_tokens)
    assert music == example_music


def test_encode_decode(
    tokenizer: MultitrackMusicTransformerTokenizer,
    example_music: muspy.Music,
):
    # NOTE: this test only works for music where all note durations are in the list of known durations
    assert tokenizer.decode(tokenizer.encode(example_music)) == example_music


def test_save_load(tmp_path, tokenizer: MultitrackMusicTransformerTokenizer):
    output_path = tmp_path / "tokenizer.json"
    tokenizer.save(output_path)
    loaded_tokenizer = MultitrackMusicTransformerTokenizer.load(output_path)
    assert loaded_tokenizer == tokenizer


def test_vocab_size(tokenizer: MultitrackMusicTransformerTokenizer):
    assert tokenizer.vocab_size == {
        "event_type": 5,
        "beat": 256 + 1,
        "position": 12 + 1,
        "pitch": 128 + 1,
        "duration": 32 + 1,
        "instrument": 64 + 1,
    }


def test_truncate_sequence(example_music: muspy.Music):
    tokenizer = MultitrackMusicTransformerTokenizer(max_seq_len=6)
    tokens = tokenizer.encode(example_music)

    token_offset = 1
    expected_codes = torch.tensor(
        [
            # event_type, beat, position, pitch, duration, instrument
            # start of song
            [0, 0, 0, 0, 0, 0],
            # instruments
            [1, 0, 0, 0, 0, 0 + token_offset],
            # start of notes
            [2, 0, 0, 0, 0, 0],
            # notes
            # duration maps to the *index* of the closest known duration
            [3, 0 + token_offset, 0 + token_offset, 60 + token_offset, 11 + token_offset, 0 + token_offset],
            [3, 0 + token_offset, 0 + token_offset, 62 + token_offset, 7 + token_offset, 0 + token_offset],
            # rest of notes get truncated
            # end of song
            [4, 0, 0, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    expected_results = {
        "event_type": expected_codes[:, 0],
        "beat": expected_codes[:, 1],
        "position": expected_codes[:, 2],
        "pitch": expected_codes[:, 3],
        "duration": expected_codes[:, 4],
        "instrument": expected_codes[:, 5],
    }
    for key in tokens:
        assert torch.all(tokens[key] == expected_results[key])
