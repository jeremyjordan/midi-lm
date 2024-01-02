from dataclasses import dataclass


@dataclass
class MultipleSequenceDictCollator:
    _target_: str = "midi_lm.collators.collate_dict_multi_seqs"
