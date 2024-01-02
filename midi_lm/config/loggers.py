from dataclasses import dataclass


@dataclass
class TensorBoardLoggerConfig:
    _target_: str = "lightning.pytorch.loggers.TensorBoardLogger"
    save_dir: str = "${hydra:run.dir}/"
    name: str = "midi-language-modeling"


@dataclass
class WeightsAndBiasesLoggerConfig:
    _target_: str = "midi_lm.loggers.FaultTolerantWandbLogger"
    project: str = "midi-language-modeling"
    log_model: bool = True
    save_dir: str = "${hydra:run.dir}/checkpoints/"


@dataclass
class WeightsAndBiasesTestLoggerConfig:
    _target_: str = "midi_lm.loggers.FaultTolerantWandbLogger"
    project: str = "midi-language-modeling-test"
    log_model: bool = True
    save_dir: str = "${hydra:run.dir}/checkpoints/"
