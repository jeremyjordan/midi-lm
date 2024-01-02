from dataclasses import dataclass


@dataclass
class MultitrackMusicTransformerTokenizerConfig:
    _target_: str = "midi_lm.tokenizers.mmt.MultitrackMusicTransformerTokenizer"
    max_seq_len: int = 1024


@dataclass
class TimeShiftPitchDurationTokenizerConfig:
    _target_: str = "midi_lm.tokenizers.tpd.TimeShiftPitchDurationTokenizer"
    max_seq_len: int = 1024


@dataclass
class StructuredSequenceTokenizerConfig:
    _target_: str = "midi_lm.tokenizers.structured_sequence.StructuredSequenceTokenizer"
    max_seq_len: int = 1024
