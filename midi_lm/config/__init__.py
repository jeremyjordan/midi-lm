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
            {"network": "mmt"},
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

config.store(group="network", name="mmt-small", node=networks.SmallMMTConfig)
config.store(group="network", name="mmt-medium", node=networks.MediumMMTConfig)
config.store(group="network", name="mmt", node=networks.MMTConfig)
config.store(group="network", name="tpd-small", node=networks.SmallTPDConfig)
config.store(group="network", name="tpd-medium", node=networks.MediumTPDConfig)
config.store(group="network", name="tpd", node=networks.TPDConfig)
config.store(group="network", name="structured-small", node=networks.SmallStructuredConfig)
config.store(group="network", name="structured-medium", node=networks.MediumStructuredConfig)
config.store(group="network", name="structured", node=networks.StructuredConfig)


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
