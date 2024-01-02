import pytest
import torch

from midi_lm.models.structured_transformer.network import StructuredTransformer, StructuredTransformerConfig


@pytest.fixture()
def example_config():
    return StructuredTransformerConfig(
        bos_token_id=1,
        eos_token_id=2,
        vocab_size=676,
        attn_layers=2,
        attn_dim=32,
        emb_dim=32,
    )


@pytest.fixture()
def example_network(example_config):
    return StructuredTransformer(example_config)


@pytest.fixture()
def example_batch():
    batch_size = 32
    seq_len = 128
    vocab_size = 676

    tokens = torch.randint(0, vocab_size, size=(batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    return {
        "token_ids": tokens,
        "attention_mask": attention_mask,
    }
