import numpy as np
import pytest

from midi_lm.models.multihead_transformer.model import LightningMultiheadTransformer


@pytest.fixture()
def lightning_model(example_config):
    return LightningMultiheadTransformer(network_config=example_config)


def test_loss_at_initialization(lightning_model, example_batch, example_vocab_sizes):
    losses = lightning_model._shared_step(example_batch)

    expected_time_shift_loss = -np.log(1 / example_vocab_sizes["time_shift"])
    expected_pitch_loss = -np.log(1 / example_vocab_sizes["pitch"])
    expected_duration_loss = -np.log(1 / example_vocab_sizes["duration"])

    time_shift_loss = losses["time_shift_loss"].item()
    pitch_loss = losses["pitch_loss"].item()
    duration_loss = losses["duration_loss"].item()

    TOLERANCE = 0.01
    assert abs(time_shift_loss - expected_time_shift_loss) / expected_time_shift_loss < TOLERANCE
    assert abs(pitch_loss - expected_pitch_loss) / expected_pitch_loss < TOLERANCE
    assert abs(duration_loss - expected_duration_loss) / expected_duration_loss < TOLERANCE


@pytest.mark.repeat(5)
def test_generation_single_step(lightning_model: LightningMultiheadTransformer):
    # default initial seed has seq_len = 1
    generated = lightning_model.generate(steps=1)

    for key in lightning_model.network.config.vocab_sizes.keys():
        assert key in generated
        assert generated[key].shape == (1, 2)


@pytest.mark.repeat(5)
def test_generation_multiple_steps(lightning_model: LightningMultiheadTransformer):
    # default initial seed has seq_len = 1
    generated = lightning_model.generate(steps=5, min_steps=5)

    for key in lightning_model.network.config.vocab_sizes.keys():
        assert key in generated
        assert generated[key].shape == (1, 6)
