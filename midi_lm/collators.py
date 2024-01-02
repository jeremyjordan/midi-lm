import torch
from torch.nn.utils.rnn import pad_sequence


def compute_sequence_mask(seq_lens: torch.Tensor):
    """Computes a boolean mask for a batch of sequences of unequal length.

    E.g. >>> compute_sequence_mask(torch.tensor([1, 2, 3])) tensor([[ True,
    False, False],
            [ True,  True, False], [ True,  True,  True]])

    Args:
        seq_lens (torch.Tensor): A tensor of shape (batch_size,) containing the
        length of each sequence in the batch.

    Returns:
        torch.Tensor: A boolean tensor of shape (batch_size, max_len) where
        max_len is the
            maximum length of the sequences in the batch.
    """
    max_len = int(seq_lens.max().item())
    range_tensor = torch.arange(max_len).expand(len(seq_lens), max_len)  # shape: (batch_size, max_len)
    lengths_tensor = seq_lens.unsqueeze(1)  # shape: (batch_size, 1)
    # broadcast comparison over the sequence length dimension
    return range_tensor < lengths_tensor


def collate_dict_multi_seqs(batch: list[dict]):
    keys = batch[0].keys()
    assert any(isinstance(batch[0][k], torch.Tensor) for k in keys)
    output = {}

    all_seq_lens = torch.tensor([], dtype=torch.long)  # Initialize with an empty tensor
    for k in keys:
        if k.startswith("_"):
            output[k] = [item[k] for item in batch]
        elif isinstance(batch[0][k], torch.Tensor):
            seq_lens = torch.tensor([len(item[k]) for item in batch], dtype=torch.long)
            if all_seq_lens.numel() > 0 and not torch.equal(seq_lens, all_seq_lens):
                raise ValueError("All sequence keys for a given row must have the same length")
            all_seq_lens = seq_lens
            output[k] = pad_sequence(
                [item[k] for item in batch],
                batch_first=True,
                padding_value=0,
            )
        else:
            print("unexpected type", type(batch[0][k]))
    output["seq_lens"] = all_seq_lens
    output["attention_mask"] = compute_sequence_mask(all_seq_lens)
    return output
