from dataclasses import dataclass


@dataclass
class AdamOptimizerConfig:
    _target_: str = "torch.optim.Adam"
    lr: float = 0.002
    weight_decay: float = 1e-3


@dataclass
class AdamWOptimizerConfig:
    _target_: str = "torch.optim.AdamW"
    lr: float = 0.002
    weight_decay: float = 1e-2


@dataclass
class SGDOptimizerConfig:
    _target_: str = "torch.optim.SGD"
    lr: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 1e-3
