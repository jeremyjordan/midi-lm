from midi_lm.models.multihead_transformer.network import (
    MultiheadTransformer,
    MultiheadTransformerConfig,
)


def test_output_shape(
    example_config: MultiheadTransformerConfig,
    example_network: MultiheadTransformer,
    example_batch: dict,
):
    batch_size = 32
    seq_len = 128

    output = example_network(example_batch)
    assert output["time_shift"].shape == (batch_size, seq_len, example_config.vocab_sizes["time_shift"])
    assert output["pitch"].shape == (batch_size, seq_len, example_config.vocab_sizes["pitch"])
    assert output["duration"].shape == (batch_size, seq_len, example_config.vocab_sizes["duration"])
