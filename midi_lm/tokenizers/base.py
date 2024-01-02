from abc import ABC, abstractmethod
from dataclasses import dataclass

import muspy
import torch


@dataclass
class TokenizerConfig:
    # special tokens
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    unk_token_id: int | None = None
    sep_token_id: int | None = None
    mask_token_id: int | None = None
    pad_token_id: int | None = 0
    special_token_offset: int = 1
    # sequence constraints
    max_seq_len: int = 1024


class BaseTokenizer(ABC):
    """Base class for tokenizers."""

    config: TokenizerConfig

    @abstractmethod
    def encode(self, music: muspy.Music) -> dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def decode(self, tokens: dict[str, torch.Tensor]) -> muspy.Music:
        pass

    @abstractmethod
    def save(self, filepath):
        pass

    @classmethod
    @abstractmethod
    def load(cls, filepath):
        pass
