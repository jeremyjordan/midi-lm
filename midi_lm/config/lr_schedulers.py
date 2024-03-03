from dataclasses import dataclass
from typing import Any


@dataclass
class ReduceLROnPlateau:
    _target_: str = "torch.optim.lr_scheduler.ReduceLROnPlateau"
    mode: str = "min"
    patience: int = 10
    min_lr: float = 0.000001
    factor: float = 0.1
    verbose: bool = True


@dataclass
class ReduceLROnPlateauConfig:
    scheduler: Any = ReduceLROnPlateau
    interval: str = "epoch"
    frequency: int = 1
    monitor: str = "val/loss"


@dataclass
class CosineAnnealingWarmRestarts:
    _target_: str = "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts"
    T_0: int = 2000
    T_mult: int = 2
    eta_min: float = 1e-6


@dataclass
class CosineAnnealingWarmRestartsConfig:
    scheduler: Any = CosineAnnealingWarmRestarts
    interval: str = "step"
    frequency: int = 1
