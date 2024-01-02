import lightning.pytorch as pl
import numpy as np
import pytest
import torch

from midi_lm.collators import collate_dict_multi_seqs
from midi_lm.datasets.eighth_notes import EighthNotesDataModule
from midi_lm.models.multitrack_music_transformer.model import LightningMultitrackMusicTransformer
from midi_lm.tokenizers.mmt import MultitrackMusicTransformerTokenizer


@pytest.fixture()
def lightning_model(example_config):
    return LightningMultitrackMusicTransformer(network_config=example_config)


def test_loss_at_initialization(lightning_model, example_batch, example_vocab_sizes):
    losses = lightning_model._shared_step(example_batch)

    expected_event_type_loss = -np.log(1 / example_vocab_sizes["event_type"])
    expected_beat_loss = -np.log(1 / example_vocab_sizes["beat"])
    expected_position_loss = -np.log(1 / example_vocab_sizes["position"])
    expected_pitch_loss = -np.log(1 / example_vocab_sizes["pitch"])
    expected_duration_loss = -np.log(1 / example_vocab_sizes["duration"])
    expected_instrument_loss = -np.log(1 / example_vocab_sizes["instrument"])

    event_type_loss = losses["event_type_loss"].item()
    beat_loss = losses["beat_loss"].item()
    position_loss = losses["position_loss"].item()
    pitch_loss = losses["pitch_loss"].item()
    duration_loss = losses["duration_loss"].item()
    instrument_loss = losses["instrument_loss"].item()

    TOLERANCE = 0.01
    assert abs(event_type_loss - expected_event_type_loss) / expected_event_type_loss < TOLERANCE
    assert abs(beat_loss - expected_beat_loss) / expected_beat_loss < TOLERANCE
    assert abs(position_loss - expected_position_loss) / expected_position_loss < TOLERANCE
    assert abs(pitch_loss - expected_pitch_loss) / expected_pitch_loss < TOLERANCE
    assert abs(duration_loss - expected_duration_loss) / expected_duration_loss < TOLERANCE
    assert abs(instrument_loss - expected_instrument_loss) / expected_instrument_loss < TOLERANCE


@pytest.mark.repeat(5)
def test_generation_single_step(lightning_model: LightningMultitrackMusicTransformer):
    # default initial seed has seq_len = 3
    generated = lightning_model.generate(steps=1)

    for key in ("event_type", "beat", "position", "pitch", "duration", "instrument"):
        assert key in generated
        assert generated[key].shape == (1, 4)


@pytest.mark.repeat(5)
def test_generation_multiple_steps(lightning_model: LightningMultitrackMusicTransformer):
    # default initial seed has seq_len = 3
    generated = lightning_model.generate(steps=5, min_steps=5, monotonic_keys=("event_type",))

    for key in ("event_type", "beat", "position", "pitch", "duration", "instrument"):
        assert key in generated
        assert generated[key].shape == (1, 8)


@pytest.mark.slow()
def test_overfit_model(lightning_model: LightningMultitrackMusicTransformer):
    tokenizer = MultitrackMusicTransformerTokenizer()
    dataset = EighthNotesDataModule(
        dataset_dir="data/eighth_notes",
        tokenizer=tokenizer,
        batch_size=1,
        collate_fn=collate_dict_multi_seqs,
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator=device,
        deterministic=True,
        logger=False,
        enable_checkpointing=False,
        overfit_batches=1,
        gradient_clip_val=1.0,
    )
    trainer.fit(model=lightning_model, datamodule=dataset)
    loss = trainer.logged_metrics["train/loss"].item()  # type: ignore
    assert loss < 0.05, "didn't successfully overfit model"
