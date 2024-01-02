from dataclasses import dataclass

import torch
import torch.nn as nn
from x_transformers.x_transformers import Decoder


@dataclass(frozen=True)
class MultiheadTransformerConfig:
    # special tokens
    bos_key: str
    eos_key: str
    bos_token_id: int
    eos_token_id: int
    # vocab sizes
    vocab_sizes: dict[str, int]
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


class MultiheadTransformer(nn.Module):
    def __init__(
        self,
        config: MultiheadTransformerConfig,
    ) -> None:
        super().__init__()
        self.config = config

        self.embeddings = nn.ModuleDict(
            {
                key: nn.Embedding(
                    num_embeddings=value,
                    embedding_dim=config.emb_dim,
                    padding_idx=config.padding_idx,
                )
                for key, value in config.vocab_sizes.items()
            }
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
        self.out = nn.ModuleDict(
            {key: nn.Linear(config.attn_dim, value) for key, value in config.vocab_sizes.items()}
        )
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
        x = torch.stack([self.embeddings[key](batch[key]) for key in self.embeddings], dim=-1).sum(dim=-1)
        # embedding output shapes: (batch_size, seq_len, emb_dim)
        x = self.emb_dropout(x)
        x = self.post_emb_norm(x)
        # x shape: (batch_size, seq_len, emb_dim)
        x = self.project_emb(x)
        # x shape: (batch_size, seq_len, attn_dim)

        # pass through the attention layers
        x = self.attn_layers(x, mask=attention_mask)
        # x shape: (batch_size, seq_len, attn_dim)
        x = self.attn_dropout(x)
        x = self.attn_norm(x)

        # compute outputs
        outputs = {key: self.out[key](x) for key in self.out}

        return outputs
