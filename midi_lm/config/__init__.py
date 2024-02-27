from dataclasses import dataclass, field
from typing import Any, List

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from midi_lm.config import (
    _internal,
    collator,
    compute,
    datasets,
    loggers,
    lr_schedulers,
    models,
    networks,
    optimizers,
    tokenizers,
    trainer,
    transforms,
)


@dataclass
class TrainingConfig:
    defaults: List[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"collator": "multi-seq-dict"},
            {"compute": "local"},
            {"dataset": "scales"},
            {"logger": "tensorboard"},
            {"lr_scheduler": "cosine"},
            {"model": "mmt"},
            {"network": "mmt-20m"},
            {"optimizer": "adamw"},
            {"tokenizer": "mmt"},
            {"trainer": "smoke-test"},
            {"transforms": "crop"},
        ]
    )
    collator: Any = MISSING
    compute: Any = MISSING
    dataset: Any = MISSING
    logger: Any = MISSING
    lr_scheduler: Any = MISSING
    model: Any = MISSING
    network: Any = MISSING
    optimizer: Any = MISSING
    tokenizer: Any = MISSING
    trainer: Any = MISSING
    transforms: Any = MISSING
    resume_from_checkpoint: str | None = None


config = ConfigStore.instance()
config.store(name="config", node=TrainingConfig)
config.store(group="hydra", name="config", node=_internal.hydra_conf, provider="hydra")

config.store(group="collator", name="multi-seq-dict", node=collator.MultipleSequenceDictCollator)

config.store(group="compute", name="local", node=compute.LocalConfig)
config.store(group="compute", name="cpu", node=compute.ModalCpuConfig)
config.store(group="compute", name="a10g", node=compute.ModalGpuA10gConfig)
config.store(group="compute", name="a100", node=compute.ModalGpuA100Config)

config.store(group="dataset", name="scales", node=datasets.ScalesDatasetConfig)
config.store(group="dataset", name="bach", node=datasets.BachChoralesDatasetConfig)
config.store(group="dataset", name="nes", node=datasets.NESDatasetConfig)
config.store(group="dataset", name="maestro", node=datasets.MaestroDatasetConfig)
config.store(group="dataset", name="symphony-net", node=datasets.SymphonyNetConfig)

config.store(group="logger", name="tensorboard", node=loggers.TensorBoardLoggerConfig)
config.store(group="logger", name="wandb", node=loggers.WeightsAndBiasesLoggerConfig)
config.store(group="logger", name="wandb-test", node=loggers.WeightsAndBiasesTestLoggerConfig)

config.store(group="lr_scheduler", name="plateau", node=lr_schedulers.ReduceLROnPlateauConfig)
config.store(group="lr_scheduler", name="cosine", node=lr_schedulers.CosineAnnealingWarmRestartsConfig)

config.store(group="model", name="mmt", node=models.MultitrackMusicTransformerModelConfig)
config.store(group="model", name="multihead-transformer", node=models.MultiheadTransformerModelConfig)
config.store(group="model", name="structured", node=models.StructuredTransformerModelConfig)

config.store(group="network", name="mmt-1m", node=networks.MultitrackMusicTransformerNetwork1M)
config.store(group="network", name="mmt-7m", node=networks.MultitrackMusicTransformerNetwork7M)
config.store(group="network", name="mmt-20m", node=networks.MultitrackMusicTransformerNetwork20M)
config.store(group="network", name="tpd-1m", node=networks.TimeshiftPitchDurationNetwork1M)
config.store(group="network", name="tpd-6m", node=networks.TimeshiftPitchDurationNetwork6M)
config.store(group="network", name="tpd-19m", node=networks.TimeshiftPitchDurationNetwork19M)
config.store(group="network", name="structured-1m", node=networks.StructuredSequenceNetwork1M)
config.store(group="network", name="structured-7m", node=networks.StructuredSequenceNetwork7M)
config.store(group="network", name="structured-20m", node=networks.StructuredSequenceNetwork20M)


config.store(group="optimizer", name="adam", node=optimizers.AdamOptimizerConfig)
config.store(group="optimizer", name="adamw", node=optimizers.AdamWOptimizerConfig)
config.store(group="optimizer", name="sgd", node=optimizers.SGDOptimizerConfig)

config.store(group="tokenizer", name="mmt", node=tokenizers.MultitrackMusicTransformerTokenizerConfig)
config.store(group="tokenizer", name="tpd", node=tokenizers.TimeShiftPitchDurationTokenizerConfig)
config.store(group="tokenizer", name="structured", node=tokenizers.StructuredSequenceTokenizerConfig)

config.store(group="trainer", name="cpu", node=trainer.CpuTrainerConfig)
config.store(group="trainer", name="gpu", node=trainer.GpuTrainerConfig)
config.store(group="trainer", name="mps", node=trainer.MpsTrainerConfig)
config.store(group="trainer", name="smoke-test", node=trainer.CpuDebugTrainerConfig)

config.store(group="transforms", name="crop", node=transforms.CropConfig)
config.store(group="transforms", name="crop-transpose", node=transforms.CropTransposeConfig)
