from dataclasses import dataclass


@dataclass
class CpuDebugTrainerConfig:
    _target_: str = "lightning.pytorch.Trainer"
    max_epochs: int = 20
    accelerator: str = "cpu"
    log_every_n_steps: int = 1
    overfit_batches: int = 1
    gradient_clip_val: float = 1.0
    val_check_interval: int | float = 10


@dataclass
class MpsTrainerConfig:
    _target_: str = "lightning.pytorch.Trainer"
    max_epochs: int = 200
    log_every_n_steps: int = 1
    accelerator: str = "mps"
    devices: int = 1
    gradient_clip_val: float = 1.0
    val_check_interval: int | float = 150


@dataclass
class CpuTrainerConfig:
    _target_: str = "lightning.pytorch.Trainer"
    max_epochs: int = 200
    accelerator: str = "cpu"
    log_every_n_steps: int = 1
    gradient_clip_val: float = 1.0
    val_check_interval: int | float = 150


@dataclass
class GpuTrainerConfig:
    _target_: str = "lightning.pytorch.Trainer"
    max_epochs: int = 200
    log_every_n_steps: int = 1
    devices: int = 1
    accelerator: str = "gpu"
    precision: str = "16-mixed"
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 4
