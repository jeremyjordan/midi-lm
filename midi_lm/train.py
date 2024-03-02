import sys
from pathlib import Path
from pprint import pformat

import hydra
import lightning.pytorch as pl
import typer
import wandb
from hydra import compose, initialize_config_module
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra.core.singleton import Singleton
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import OmegaConf, open_dict

from midi_lm import logger
from midi_lm.callbacks import GenerateSequenceCallback
from midi_lm.config import TrainingConfig
from midi_lm.config.transforms import create_transforms
from midi_lm.modal_config import stub, train_a10g, train_a100, train_cpu
from midi_lm.tokenizers import BaseTokenizer

resume_cli = typer.Typer()


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

    logger = hydra.utils.instantiate(config.logger)
    if isinstance(logger, WandbLogger):
        # hack to pass through wandb settings since a pure dict doesn't seem to work and hydra instantiate
        # won't allow passing through a dataclass (which is what wandb.Settings resolves to)
        logger._wandb_init.update({"settings": wandb.Settings(disable_job_creation=True)})
        logger.watch(model, log="gradients", log_freq=100, log_graph=False)

    # if we're resuming a training run, download the checkpoint artifact
    if config.resume_from_checkpoint is not None:
        api = wandb.Api()
        model_checkpoint = api.artifact(config.resume_from_checkpoint)
        ckpt_path = str(model_checkpoint.file(root=checkpoint_dir.as_posix()))
    else:
        ckpt_path = None

    if config.resume_from_checkpoint and isinstance(logger, WandbLogger):
        logger.experiment.use_artifact(config.resume_from_checkpoint)

    trainer: pl.Trainer = hydra.utils.instantiate(config.trainer, logger=logger, callbacks=callbacks)
    logger.log_hyperparams(OmegaConf.to_container(config))  # type: ignore
    trainer.fit(model=model, datamodule=dataset, ckpt_path=ckpt_path)


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


@resume_cli.command()
def resume(
    artifact_name: str = typer.Argument(
        help="Name of the wandb artifact to resume training from (e.g. 'user/project/model:v0')"
    ),
    hydra_config_overrides: list[str] = typer.Argument(  # noqa: B008
        None,
        help="Optional list of Hydra overrides to update the configuration from the initial training run.",
    ),
) -> None:
    api = wandb.Api()
    logger.info(f"Querying wandb for artifact {artifact_name}")
    model_checkpoint = api.artifact(artifact_name)
    run = model_checkpoint.logged_by()
    if run:
        logger.info(f"Resuming training from run: {run.id}")
        original_run_config = OmegaConf.create(run.config)

        # initialize hydra config module and config store singleton
        initialize_config_module(config_module="midi_lm.config", version_base=None)
        cs = ConfigStore.instance()
        # insert the original run config into the config store
        cs.store(name="_config", node=original_run_config)
        # apply the overrides to the original run config
        # return the hydra config to update the global state
        resume_config = compose(
            config_name="_config",
            overrides=hydra_config_overrides,
            return_hydra_config=True,
        )
        # update the global state to mirror the logic in the hydra main function
        # https://github.com/facebookresearch/hydra/issues/2017#issuecomment-1254220345
        HydraConfig.instance().set_config(resume_config)
        # after saving the hydra config to the global state, we can remove it from our training config
        # in order to modify the DictConfig, we need to use the open_dict context manager
        # https://omegaconf.readthedocs.io/en/latest/usage.html#struct-flag
        with open_dict(resume_config):
            resume_config.pop("hydra")

        resume_config.resume_from_checkpoint = artifact_name
        logger.debug(f"Resuming training with config:\n{pformat(OmegaConf.to_container(resume_config))}\n")
        train(resume_config)
    else:
        raise ValueError(f"Could not find run for artifact {artifact_name}")


if __name__ == "__main__":
    train()
