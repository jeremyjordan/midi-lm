import sys
from pathlib import Path

import hydra
import lightning.pytorch as pl
from hydra.core.hydra_config import HydraConfig
from hydra.core.singleton import Singleton
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import OmegaConf

from midi_lm.callbacks import GenerateSequenceCallback
from midi_lm.config import TrainingConfig
from midi_lm.config.transforms import create_transforms
from midi_lm.modal import remote_image, stub, volume
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


# wrapper for remote execution
def train_remote(config: TrainingConfig, singleton_state: dict, args: list[str]):
    import os

    from hydra.core.utils import setup_globals

    # perform some initial hydra setup
    setup_globals()
    Singleton.set_state(singleton_state)
    # set environment variable to change command shown in wandb run
    os.environ["WANDB_PROGRAM"] = " ".join(args)
    # use the code below once wandb fixes a bug
    # https://github.com/wandb/wandb/issues/4791
    # os.environ["WANDB_PROGRAM"] = args[0]
    # os.environ["WANDB_ARGS"] = json.dumps(args[1:])

    # execute the model training
    run_training(config)


# CLI entrypoint
@hydra.main(version_base=None, config_name="config")
def train(config: TrainingConfig) -> None:
    if config.compute.local:
        print("Running training locally")
        run_training(config)
    else:
        print("Running remote training")
        fn = stub.function(
            image=remote_image,
            volumes={"/root/data": volume},
            timeout=config.compute.timeout,
            cpu=config.compute.cpu,
            memory=config.compute.memory,
            gpu=config.compute.gpu,
        )(train_remote)
        with stub.run(detach=True, show_progress=False):
            # copy hydra config state and pass to remote execution
            singleton_state = Singleton.get_state()
            fn.remote(config, singleton_state, args=sys.argv)


if __name__ == "__main__":
    train()
