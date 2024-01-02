from dataclasses import dataclass


@dataclass
class MultitrackMusicTransformerModelConfig:
    _target_: str = "midi_lm.models.multitrack_music_transformer.model.LightningMultitrackMusicTransformer"


@dataclass
class MultiheadTransformerModelConfig:
    _target_: str = "midi_lm.models.multihead_transformer.model.LightningMultiheadTransformer"


@dataclass
class StructuredTransformerModelConfig:
    _target_: str = "midi_lm.models.structured_transformer.model.LightningStructuredTransformer"
