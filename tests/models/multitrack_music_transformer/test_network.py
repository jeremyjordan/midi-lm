from midi_lm.models.multitrack_music_transformer.network import (
    MultitrackMusicTransformer,
    MultitrackMusicTransformerConfig,
)


def test_output_shape(
    example_config: MultitrackMusicTransformerConfig,
    example_network: MultitrackMusicTransformer,
    example_batch: dict,
):
    batch_size = 32
    seq_len = 128

    output = example_network(example_batch)
    assert output["event_type"].shape == (batch_size, seq_len, example_config.event_type_dim)
    assert output["beat"].shape == (batch_size, seq_len, example_config.beat_dim)
    assert output["position"].shape == (batch_size, seq_len, example_config.position_dim)
    assert output["pitch"].shape == (batch_size, seq_len, example_config.pitch_dim)
    assert output["duration"].shape == (batch_size, seq_len, example_config.duration_dim)
    assert output["instrument"].shape == (batch_size, seq_len, example_config.instrument_dim)
