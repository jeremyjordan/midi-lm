from dataclasses import dataclass


@dataclass
class ScalesDatasetConfig:
    _target_: str = "midi_lm.datasets.scales.ScalesDataModule"
    dataset_dir: str = "data/scales/"
    batch_size: int = 8
    num_workers: int = 0  # small dataset, keep on main process for much faster processing


@dataclass
class BachChoralesDatasetConfig:
    _target_: str = "midi_lm.datasets.bach_chorales.BachChoralesDataModule"
    dataset_dir: str = "data/bach_chorales/"
    batch_size: int = 16
    num_workers: int = 0


@dataclass
class NESDatasetConfig:
    _target_: str = "midi_lm.datasets.nes.NESDataModule"
    dataset_dir: str = "data/nes/"
    batch_size: int = 16
    num_workers: int = 0


@dataclass
class MaestroDatasetConfig:
    _target_: str = "midi_lm.datasets.maestro.MaestroDataModule"
    dataset_dir: str = "data/maestro/"
    batch_size: int = 16
    num_workers: int = 4


@dataclass
class SymphonyNetConfig:
    _target_: str = "midi_lm.datasets.symphony_net.SymphonyNetDataModule"
    dataset_dir: str = "data/symphony_net/"
    batch_size: int = 16
    num_workers: int = 4
