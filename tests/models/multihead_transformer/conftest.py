import pytest
import torch

from midi_lm.models.multihead_transformer.network import MultiheadTransformer, MultiheadTransformerConfig


@pytest.fixture()
def example_vocab_sizes():
    return {
        "time_shift": 4,
        "pitch": 128,
        "duration": 12,
    }


@pytest.fixture()
def example_config(example_vocab_sizes):
    return MultiheadTransformerConfig(
        bos_key="pitch",
        bos_token_id=1,
        eos_key="pitch",
        eos_token_id=2,
        vocab_sizes=example_vocab_sizes,
        attn_layers=2,
        attn_dim=32,
        emb_dim=32,
    )


@pytest.fixture()
def example_network(example_config):
    return MultiheadTransformer(example_config)


@pytest.fixture()
def example_batch(example_vocab_sizes):
    batch_size = 32
    seq_len = 128

    time_shift = torch.randint(0, example_vocab_sizes["time_shift"], size=(batch_size, seq_len))
    pitch = torch.randint(0, example_vocab_sizes["pitch"], size=(batch_size, seq_len))
    duration = torch.randint(0, example_vocab_sizes["duration"], size=(batch_size, seq_len))

    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    return {
        "time_shift": time_shift,
        "pitch": pitch,
        "duration": duration,
        "attention_mask": attention_mask,
    }
