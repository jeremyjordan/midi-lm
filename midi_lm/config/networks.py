from dataclasses import dataclass, field

from midi_lm.models.multihead_transformer.network import MultiheadTransformerConfig
from midi_lm.models.multitrack_music_transformer.network import MultitrackMusicTransformerConfig
from midi_lm.models.structured_transformer.network import StructuredTransformerConfig

# ----- Multitrack Music Transformer -----


@dataclass(frozen=True)
class MultitrackMusicTransformerNetwork1M(MultitrackMusicTransformerConfig):
    _target_: str = "midi_lm.models.multitrack_music_transformer.network.MultitrackMusicTransformerConfig"
    # embedding parameters
    emb_dim: int = 128
    emb_dropout: float = 0.0
    l2norm_embed: bool = True
    post_emb_norm: bool = True
    # transformer parameters
    attn_dim: int = 128
    attn_layers: int = 3
    attn_heads: int = 4
    attn_dropout: float = 0.0


@dataclass(frozen=True)
class MultitrackMusicTransformerNetwork7M(MultitrackMusicTransformerConfig):
    _target_: str = "midi_lm.models.multitrack_music_transformer.network.MultitrackMusicTransformerConfig"
    # embedding parameters
    emb_dim: int = 256
    emb_dropout: float = 0.0
    l2norm_embed: bool = True
    post_emb_norm: bool = True
    # transformer parameters
    attn_dim: int = 256
    attn_layers: int = 6
    attn_heads: int = 8
    attn_dropout: float = 0.0


@dataclass(frozen=True)
class MultitrackMusicTransformerNetwork20M(MultitrackMusicTransformerConfig):
    _target_: str = "midi_lm.models.multitrack_music_transformer.network.MultitrackMusicTransformerConfig"
    # embedding parameters
    emb_dim: int = 512
    emb_dropout: float = 0.0
    l2norm_embed: bool = True
    post_emb_norm: bool = True
    # transformer parameters
    attn_dim: int = 512
    attn_layers: int = 6
    attn_heads: int = 8
    attn_dropout: float = 0.0


# ----- Multihead Transformer -----
@dataclass(frozen=True)
class TimeshiftPitchDurationNetwork1M(MultiheadTransformerConfig):
    _target_: str = "midi_lm.models.multihead_transformer.network.MultiheadTransformerConfig"
    bos_key: str = "pitch"
    bos_token_id: int = 1
    eos_key: str = "pitch"
    eos_token_id: int = 2
    vocab_sizes: dict[str, int] = field(
        default_factory=lambda: {"time_shift": 9 + 1, "pitch": 128 + 3, "duration": 13 + 1}
    )
    # embedding parameters
    emb_dim: int = 128
    emb_dropout: float = 0.0
    post_emb_norm: bool = False
    # transformer parameters
    attn_dim: int = 128
    attn_layers: int = 3
    attn_heads: int = 4
    attn_dropout: float = 0.0


@dataclass(frozen=True)
class TimeshiftPitchDurationNetwork6M(MultiheadTransformerConfig):
    _target_: str = "midi_lm.models.multihead_transformer.network.MultiheadTransformerConfig"
    bos_key: str = "pitch"
    bos_token_id: int = 1
    eos_key: str = "pitch"
    eos_token_id: int = 2
    vocab_sizes: dict[str, int] = field(
        default_factory=lambda: {"time_shift": 9 + 1, "pitch": 128 + 3, "duration": 13 + 1}
    )
    # embedding parameters
    emb_dim: int = 256
    emb_dropout: float = 0.0
    post_emb_norm: bool = False
    # transformer parameters
    attn_dim: int = 256
    attn_layers: int = 6
    attn_heads: int = 8
    attn_dropout: float = 0.0


@dataclass(frozen=True)
class TimeshiftPitchDurationNetwork19M(MultiheadTransformerConfig):
    _target_: str = "midi_lm.models.multihead_transformer.network.MultiheadTransformerConfig"
    bos_key: str = "pitch"
    bos_token_id: int = 1
    eos_key: str = "pitch"
    eos_token_id: int = 2
    vocab_sizes: dict[str, int] = field(
        default_factory=lambda: {"time_shift": 9 + 1, "pitch": 128 + 3, "duration": 13 + 1}
    )
    # embedding parameters
    emb_dim: int = 512
    emb_dropout: float = 0.0
    post_emb_norm: bool = False
    # transformer parameters
    attn_dim: int = 512
    attn_layers: int = 6
    attn_heads: int = 8
    attn_dropout: float = 0.0


# ----- Structured Transformer -----
@dataclass(frozen=True)
class StructuredSequenceNetwork1M(StructuredTransformerConfig):
    _target_: str = "midi_lm.models.structured_transformer.network.StructuredTransformerConfig"
    # embedding parameters
    emb_dim: int = 128
    emb_dropout: float = 0
    post_emb_norm: bool = False
    # transformer parameters
    attn_dim: int = 128
    attn_layers: int = 3
    attn_heads: int = 4
    attn_dropout: float = 0


@dataclass(frozen=True)
class StructuredSequenceNetwork7M(StructuredTransformerConfig):
    _target_: str = "midi_lm.models.structured_transformer.network.StructuredTransformerConfig"
    # embedding parameters
    emb_dim: int = 256
    emb_dropout: float = 0.0
    post_emb_norm: bool = False
    # transformer parameters
    attn_dim: int = 256
    attn_layers: int = 6
    attn_heads: int = 8
    attn_dropout: float = 0.0


@dataclass(frozen=True)
class StructuredSequenceNetwork20M(StructuredTransformerConfig):
    _target_: str = "midi_lm.models.structured_transformer.network.StructuredTransformerConfig"
    # embedding parameters
    emb_dim: int = 512
    emb_dropout: float = 0.0
    post_emb_norm: bool = False
    # transformer parameters
    attn_dim: int = 512
    attn_layers: int = 6
    attn_heads: int = 8
    attn_dropout: float = 0.0
