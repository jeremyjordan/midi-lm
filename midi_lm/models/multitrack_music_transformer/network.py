"""
Description from the paper: https://arxiv.org/abs/2207.06983

We base the proposed model on a decoder-only transformer model [25, 26]. Unlike
a standard transformer model, whose inputs and outputs are one-dimensional, the
proposed model has multi-dimensional input and output spaces similar to [6], as
illustrated in Figure 2. The model is trained to minimize the sum of the cross
entropy losses of different fields under an autoregressive setting. We adopt a
learnable absolute positional embedding [3].

For the proposed MMT model, we use 6 transformer decoder blocks, with a model
dimension of 512 and 8 self-attention heads. All input embeddings have 512
dimensions. We trim the code sequences to a maximum length of 1,024 and a
maximum beat of 256.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from x_transformers.x_transformers import AbsolutePositionalEmbedding, Decoder

from midi_lm.models.multitrack_music_transformer.const import (
    DEFAULT_KNOWN_DURATIONS,
    DEFAULT_MAX_BEATS,
    DEFAULT_RESOLUTION,
)


@dataclass(frozen=True)
class MultitrackMusicTransformerConfig:
    max_seq_len: int = 1024
    padding_idx: int = 0
    # vocab sizes
    event_type_dim: int = 5
    beat_dim: int = DEFAULT_MAX_BEATS + 1
    position_dim: int = DEFAULT_RESOLUTION + 1
    pitch_dim: int = 128 + 1
    duration_dim: int = len(DEFAULT_KNOWN_DURATIONS) + 1
    instrument_dim: int = 128 + 1
    # embedding parameters
    emb_dim: int = 512
    emb_dropout: float = 0.0
    l2norm_embed: bool = False
    post_emb_norm: bool = False
    # transformer parameters
    attn_dim: int = 512
    attn_layers: int = 6
    attn_heads: int = 8
    attn_dropout: float = 0.0


class MultitrackMusicTransformer(nn.Module):
    def __init__(
        self,
        config: MultitrackMusicTransformerConfig,
    ) -> None:
        super().__init__()
        self.config = config

        # token embeddings
        self.event_type_emb = nn.Embedding(
            num_embeddings=config.event_type_dim,
            embedding_dim=config.emb_dim,
            padding_idx=config.padding_idx,
        )
        self.beat_emb = nn.Embedding(
            num_embeddings=config.beat_dim, embedding_dim=config.emb_dim, padding_idx=config.padding_idx
        )
        self.position_emb = nn.Embedding(
            num_embeddings=config.position_dim, embedding_dim=config.emb_dim, padding_idx=config.padding_idx
        )
        self.pitch_emb = nn.Embedding(
            num_embeddings=config.pitch_dim, embedding_dim=config.emb_dim, padding_idx=config.padding_idx
        )
        self.duration_emb = nn.Embedding(
            num_embeddings=config.duration_dim, embedding_dim=config.emb_dim, padding_idx=config.padding_idx
        )
        self.instrument_emb = nn.Embedding(
            num_embeddings=config.instrument_dim,
            embedding_dim=config.emb_dim,
            padding_idx=config.padding_idx,
        )

        # positional embeddings
        self.seq_position_emb = AbsolutePositionalEmbedding(
            config.emb_dim, config.max_seq_len, l2norm_embed=config.l2norm_embed
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
        self.attn_layers = Decoder(dim=config.attn_dim, depth=config.attn_layers, heads=config.attn_heads)
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.attn_norm = nn.LayerNorm(config.attn_dim)

        # output
        self.event_type_out = nn.Linear(config.attn_dim, config.event_type_dim)
        self.beat_out = nn.Linear(config.attn_dim, config.beat_dim)
        self.position_out = nn.Linear(config.attn_dim, config.position_dim)
        self.pitch_out = nn.Linear(config.attn_dim, config.pitch_dim)
        self.duration_out = nn.Linear(config.attn_dim, config.duration_dim)
        self.instrument_out = nn.Linear(config.attn_dim, config.instrument_dim)

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
        # token inputs shape: (batch_size, seq_len)
        event_type = batch["event_type"]
        beat = batch["beat"]
        position = batch["position"]
        pitch = batch["pitch"]
        duration = batch["duration"]
        instrument = batch["instrument"]
        # attention_mask shape: (batch_size, seq_len)
        attention_mask = batch["attention_mask"]

        # token embeddings
        # token input shapes: (batch_size, seq_len)
        # embedding output shapes: (batch_size, seq_len, emb_dim)
        event_type_emb = self.event_type_emb(event_type)
        beat_emb = self.beat_emb(beat)
        position_emb = self.position_emb(position)
        pitch_emb = self.pitch_emb(pitch)
        duration_emb = self.duration_emb(duration)
        instrument_emb = self.instrument_emb(instrument)

        # positional embeddings
        # the position embedding module needs to see the inputs
        # in order to determine the sequence length and device
        # inputs shape: (batch_size, seq_len, feature_dim)
        inputs = torch.stack((event_type, beat, position, pitch, duration, instrument), dim=2)
        seq_position_emb = self.seq_position_emb(inputs)
        # seq_position_emb shapes: (seq_len, emb_dim)

        # combine embeddings
        # seq_position_emb gets broadcasted over the batch dimension
        x = (
            event_type_emb
            + beat_emb
            + position_emb
            + pitch_emb
            + duration_emb
            + seq_position_emb
            + instrument_emb
        )
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
        # output shapes: (batch_size, seq_len, num_classes)
        event_type_out = self.event_type_out(x)
        beat_out = self.beat_out(x)
        position_out = self.position_out(x)
        pitch_out = self.pitch_out(x)
        duration_out = self.duration_out(x)
        instrument_out = self.instrument_out(x)

        return {
            "event_type": event_type_out,
            "beat": beat_out,
            "position": position_out,
            "pitch": pitch_out,
            "duration": duration_out,
            "instrument": instrument_out,
        }
