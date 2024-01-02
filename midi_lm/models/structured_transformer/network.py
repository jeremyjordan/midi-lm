from dataclasses import dataclass

import torch
import torch.nn as nn
from x_transformers.x_transformers import Decoder


@dataclass(frozen=True)
class StructuredTransformerConfig:
    # special tokens
    bos_token_id: int = 1
    eos_token_id: int = 2
    # vocab size
    vocab_size: int = 676
    special_token_range: tuple[int, int] = (0, 3)
    pitch_range: tuple[int, int] = (3, 131)
    velocity_range: tuple[int, int] = (131, 163)
    duration_range: tuple[int, int] = (163, 419)
    timeshift_range: tuple[int, int] = (419, 676)
    # sequence parameters
    max_seq_len: int = 1024
    padding_idx: int = 0
    # embedding parameters
    emb_dim: int = 512
    emb_dropout: float = 0.0
    post_emb_norm: bool = False
    # transformer parameters
    attn_dim: int = 512
    attn_layers: int = 6
    attn_heads: int = 8
    attn_dropout: float = 0.0


class StructuredTransformer(nn.Module):
    def __init__(
        self,
        config: StructuredTransformerConfig,
    ) -> None:
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.emb_dim,
            padding_idx=config.padding_idx,
        )

        # normalization and dropout
        self.post_emb_norm = nn.LayerNorm(config.emb_dim) if config.post_emb_norm else nn.Identity()
        self.emb_dropout = nn.Dropout(config.emb_dropout)
        self.project_emb = (
            nn.Linear(config.emb_dim, config.attn_dim)
            if config.emb_dim != config.attn_dim
            else nn.Identity()
        )

        # transformer
        self.attn_layers = Decoder(
            dim=config.attn_dim,
            depth=config.attn_layers,
            heads=config.attn_heads,
            alibi_pos_bias=True,
        )
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.attn_norm = nn.LayerNorm(config.attn_dim)

        # output
        self.out = nn.Linear(config.attn_dim, config.vocab_size)

        # token type masks
        pitch_mask = torch.ones((self.config.vocab_size), dtype=torch.bool)
        pitch_mask[self.config.pitch_range[0] : self.config.pitch_range[1] + 1] = False
        velocity_mask = torch.ones((self.config.vocab_size), dtype=torch.bool)
        velocity_mask[self.config.velocity_range[0] : self.config.velocity_range[1] + 1] = False
        duration_mask = torch.ones((self.config.vocab_size), dtype=torch.bool)
        duration_mask[self.config.duration_range[0] : self.config.duration_range[1] + 1] = False
        time_shift_mask = torch.ones((self.config.vocab_size), dtype=torch.bool)
        time_shift_mask[self.config.timeshift_range[0] : self.config.timeshift_range[1] + 1] = False
        # allow special tokens after structured sequence is complete
        time_shift_mask[self.config.special_token_range[0] : self.config.special_token_range[1] + 1] = False

        self.register_buffer("pitch_mask", pitch_mask)
        self.register_buffer("velocity_mask", velocity_mask)
        self.register_buffer("duration_mask", duration_mask)
        self.register_buffer("time_shift_mask", time_shift_mask)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, batch):
        # attention_mask shape: (batch_size, seq_len)
        attention_mask = batch["attention_mask"]

        # token input shapes: (batch_size, seq_len)
        x = batch["token_ids"]
        x = self.embedding(x)
        # shape: (batch_size, seq_len, emb_dim)
        x = self.emb_dropout(x)
        x = self.post_emb_norm(x)
        # shape: (batch_size, seq_len, emb_dim)
        x = self.project_emb(x)
        # shape: (batch_size, seq_len, attn_dim)

        # pass through the attention layers
        x = self.attn_layers(x, mask=attention_mask)
        # shape: (batch_size, seq_len, attn_dim)
        x = self.attn_dropout(x)
        x = self.attn_norm(x)

        # compute outputs
        output = self.out(x)
        # shape: (batch_size, seq_len, vocab_size)
        return output
