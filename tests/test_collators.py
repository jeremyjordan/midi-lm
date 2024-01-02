import pytest
import torch

from midi_lm.collators import collate_dict_multi_seqs, compute_sequence_mask


def assert_batch_equal(batch, expected_output):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            assert torch.equal(v, expected_output[k]), f"key: {k} failed"
        else:
            assert v == expected_output[k], f"key: {k} failed"


def test_homogenous_batch():
    batch = [
        {"seq_a": torch.tensor([1, 2, 3]), "seq_b": torch.tensor([0, 1, 0])},
        {"seq_a": torch.tensor([4, 5, 6]), "seq_b": torch.tensor([1, 0, 1])},
        {"seq_a": torch.tensor([7, 8, 9]), "seq_b": torch.tensor([0, 1, 1])},
    ]
    expected_output = {
        "seq_a": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.long),
        "seq_b": torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=torch.long),
        "seq_lens": torch.tensor([3, 3, 3], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=torch.bool),
    }
    result = collate_dict_multi_seqs(batch)
    assert_batch_equal(result, expected_output)


def test_heterogenous_batch():
    batch = [
        {"seq_a": torch.tensor([1, 2, 3]), "seq_b": torch.tensor([0, 1, 0])},
        {"seq_a": torch.tensor([4, 5, 6, 7]), "seq_b": torch.tensor([1, 0, 0, 1])},
        {"seq_a": torch.tensor([7, 8, 9]), "seq_b": torch.tensor([0, 1, 0])},
    ]
    expected_output = {
        "seq_a": torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7], [7, 8, 9, 0]], dtype=torch.long),
        "seq_b": torch.tensor([[0, 1, 0, 0], [1, 0, 0, 1], [0, 1, 0, 0]], dtype=torch.long),
        "seq_lens": torch.tensor([3, 4, 3], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool),
    }
    result = collate_dict_multi_seqs(batch)
    assert_batch_equal(result, expected_output)


def test_inconsistent_len():
    batch = [
        {"seq_a": torch.tensor([1, 2, 3]), "seq_b": torch.tensor([0, 1])},
    ]
    with pytest.raises(ValueError, match="All sequence keys for a given row must have the same length"):
        collate_dict_multi_seqs(batch)


def test_compute_sequence_mask():
    seq_lens = torch.tensor([1, 2, 3])
    expected_output = torch.tensor([[True, False, False], [True, True, False], [True, True, True]])
    result = compute_sequence_mask(seq_lens)
    assert torch.equal(result, expected_output)
