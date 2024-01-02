import pytest
import torch

from midi_lm.models.multitrack_music_transformer.network import (
    MultitrackMusicTransformer,
    MultitrackMusicTransformerConfig,
)


@pytest.fixture()
def example_vocab_sizes():
    return {
        "event_type": 5,
        "beat": 24,
        "position": 12,
        "pitch": 128,
        "duration": 12,
        "instrument": 65,
    }


@pytest.fixture()
def example_config(example_vocab_sizes):
    return MultitrackMusicTransformerConfig(
        event_type_dim=example_vocab_sizes["event_type"],
        beat_dim=example_vocab_sizes["beat"],
        position_dim=example_vocab_sizes["position"],
        pitch_dim=example_vocab_sizes["pitch"],
        duration_dim=example_vocab_sizes["duration"],
        instrument_dim=example_vocab_sizes["instrument"],
        attn_layers=4,
        attn_dim=64,
        emb_dim=64,
    )


@pytest.fixture()
def example_network(example_config):
    return MultitrackMusicTransformer(example_config)


@pytest.fixture()
def example_batch(example_vocab_sizes):
    batch_size = 32
    seq_len = 128

    event_type = torch.randint(0, example_vocab_sizes["event_type"], size=(batch_size, seq_len))
    beat = torch.randint(0, example_vocab_sizes["beat"], size=(batch_size, seq_len))
    position = torch.randint(0, example_vocab_sizes["position"], size=(batch_size, seq_len))
    pitch = torch.randint(0, example_vocab_sizes["pitch"], size=(batch_size, seq_len))
    duration = torch.randint(0, example_vocab_sizes["duration"], size=(batch_size, seq_len))
    instrument = torch.randint(0, example_vocab_sizes["instrument"], size=(batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    return {
        "event_type": event_type,
        "beat": beat,
        "position": position,
        "pitch": pitch,
        "duration": duration,
        "instrument": instrument,
        "attention_mask": attention_mask,
    }
