from midi_lm.models.structured_transformer.network import StructuredTransformer, StructuredTransformerConfig


def test_output_shape(
    example_config: StructuredTransformerConfig,
    example_network: StructuredTransformer,
    example_batch: dict,
):
    batch_size = 32
    seq_len = 128

    output = example_network(example_batch)
    assert output.shape == (batch_size, seq_len, example_config.vocab_size)
