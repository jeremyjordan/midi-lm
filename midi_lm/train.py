import sys
from pathlib import Path

import hydra
import lightning.pytorch as pl
from hydra.core.hydra_config import HydraConfig
from hydra.core.singleton import Singleton
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import OmegaConf

from midi_lm import logger
from midi_lm.callbacks import GenerateSequenceCallback
from midi_lm.config import TrainingConfig
from midi_lm.config.transforms import create_transforms
from midi_lm.modal_config import stub, train_a10g, train_a100, train_cpu
from midi_lm.tokenizers import BaseTokenizer


# underlying training function
def run_training(config: TrainingConfig):
    run_dir = HydraConfig.get().run.dir
    checkpoint_dir = Path(run_dir, "checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tokenizer: BaseTokenizer = hydra.utils.instantiate(config.tokenizer)
    collate_fn = hydra.utils.get_method(config.collator._target_)
    transforms = create_transforms(config.transforms)
    dataset: pl.LightningDataModule = hydra.utils.instantiate(
        config.dataset,
        tokenizer=tokenizer,
        collate_fn=collate_fn,
        transforms=transforms,
    )

    model: pl.LightningModule = hydra.utils.instantiate(
        config=config.model,
        network_config=config.network,
        optimizer_config=config.optimizer,
        lr_scheduler_config=config.lr_scheduler,
        _recursive_=False,
    )

    logger = hydra.utils.instantiate(config.logger)
    if isinstance(logger, WandbLogger):
        logger.watch(model, log="all", log_freq=100)

    # TODO: make these callbacks configurable
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        GenerateSequenceCallback(tokenizer=tokenizer),
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            save_top_k=3,
            monitor="train/loss",
        ),
    ]
    if config.compute.local:
        # for some reason, modal doesn't seem to show the rich progress bar
        callbacks.append(RichProgressBar())

    trainer: pl.Trainer = hydra.utils.instantiate(config.trainer, logger=logger, callbacks=callbacks)
    logger.log_hyperparams(OmegaConf.to_container(config))  # type: ignore
    trainer.fit(model=model, datamodule=dataset)


# CLI entrypoint
@hydra.main(version_base=None, config_name="config")
def train(config: TrainingConfig) -> None:
    if config.compute.local:
        logger.info("Running training locally")
        run_training(config)
    else:
        logger.info("Running remote training")
        options = {
            "cpu": train_cpu,
            "a10g": train_a10g,
            "a100": train_a100,
        }
        remote_fn = options[config.compute.hardware]
        with stub.run(detach=True, show_progress=False):
            # copy hydra config state and pass to remote execution
            singleton_state = Singleton.get_state()
            remote_fn.remote(config, singleton_state, args=sys.argv)


if __name__ == "__main__":
    train()
