import numpy as np
import pytest

from midi_lm.models.structured_transformer.model import LightningStructuredTransformer


@pytest.fixture()
def lightning_model(example_config):
    return LightningStructuredTransformer(network_config=example_config)


def test_loss_at_initialization(lightning_model, example_batch, example_config):
    output = lightning_model._shared_step(example_batch)

    expected_loss = -np.log(1 / example_config.vocab_size)
    actual_loss = output["loss"].item()

    TOLERANCE = 0.01
    assert abs(actual_loss - expected_loss) / expected_loss < TOLERANCE


@pytest.mark.repeat(5)
def test_generation_single_step(lightning_model: LightningStructuredTransformer):
    # default initial seed has seq_len = 1
    generated = lightning_model.generate(steps=1)
    assert generated["token_ids"].shape == (1, 2)


@pytest.mark.repeat(5)
def test_generation_multiple_steps(lightning_model: LightningStructuredTransformer):
    # default initial seed has seq_len = 1
    generated = lightning_model.generate(steps=5, min_steps=5)
    assert generated["token_ids"].shape == (1, 6)
