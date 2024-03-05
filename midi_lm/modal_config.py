"""
Common commands for volumes:

    modal volume create datasets-vol
    modal volume put datasets-vol data/ 
    modal volume ls datasets-vol
    modal volume rm datasets-vol -r scales/
    modal volume ls debug-vol

    modal shell midi_lm/modal_config.py::stub.train_cpu
"""

import modal

from midi_lm import ROOT_DIR
from midi_lm.config import TrainingConfig

requirements_file = ROOT_DIR.parent / "requirements.txt"

remote_image = (
    modal.Image.debian_slim(python_version="3.11.2")
    .apt_install("git")
    .pip_install_from_requirements(requirements_file.as_posix())
    .pip_install("gdown")
)

volume = modal.Volume.persisted("datasets-vol")
debug_vol = modal.Volume.persisted("debug-vol")

stub = modal.Stub(
    name="midi-language-modeling",
    image=remote_image,
    secrets=[
        modal.Secret.from_name("wandb-secret"),
    ],
    volumes={"/root/data": volume, "/root/debug": debug_vol},
)


def train_remote(config: TrainingConfig, singleton_state: dict, args: list[str]):
    import os

    import torch
    from hydra.core.singleton import Singleton
    from hydra.core.utils import setup_globals

    from midi_lm.train import run_training

    # perform some initial hydra setup
    setup_globals()
    Singleton.set_state(singleton_state)
    # set environment variable to change command shown in wandb run
    os.environ["WANDB_PROGRAM"] = " ".join(args)
    # use the code below once wandb fixes a bug
    # https://github.com/wandb/wandb/issues/4791
    # os.environ["WANDB_PROGRAM"] = args[0]
    # os.environ["WANDB_ARGS"] = json.dumps(args[1:])

    torch.set_float32_matmul_precision("medium")

    debug_vol.reload()
    volume.reload()
    # execute the model training
    run_training(config)
    debug_vol.commit()


@stub.function(
    cpu=8,
    memory=3814,  # 4gb
    timeout=60 * 60 * 12,
    volumes={"/root/data": volume, "/root/debug": debug_vol},
)
def train_cpu(config: TrainingConfig, singleton_state: dict, args: list[str]):
    train_remote(config, singleton_state, args)


@stub.function(
    cpu=16,
    memory=3814,  # 4gb
    gpu="a10g",
    timeout=60 * 60 * 12,
)
def train_a10g(config: TrainingConfig, singleton_state: dict, args: list[str]):
    train_remote(config, singleton_state, args)


@stub.function(
    cpu=16,
    memory=7628,  # 8gb
    gpu="a100",
    timeout=60 * 60 * 12,
)
def train_a100(config: TrainingConfig, singleton_state: dict, args: list[str]):
    train_remote(config, singleton_state, args)


@stub.function(
    cpu=16,
    memory=15258,  # 16gb
    gpu="h100",
    timeout=60 * 60 * 4,
)
def train_h100(config: TrainingConfig, singleton_state: dict, args: list[str]):
    train_remote(config, singleton_state, args)


# remote dataset download functions
@stub.function(volumes={"/root/data": volume})
def download_scales():
    """
    modal run midi_lm/modal_config.py::stub.download_scales
    """
    from midi_lm.datasets.scales import ScalesDataset

    ScalesDataset.download("/root/data/scales")
    volume.commit()
    ScalesDataset.make_splits("/root/data/scales")
    volume.commit()


@stub.function(volumes={"/root/data": volume})
def download_bach():
    """
    modal run midi_lm/modal_config.py::stub.download_bach
    """
    from midi_lm.datasets.bach_chorales import BachChoralesDataset

    BachChoralesDataset.download("/root/data/bach_chorales")
    volume.commit()
    BachChoralesDataset.make_splits("/root/data/bach_chorales")
    volume.commit()


@stub.function(volumes={"/root/data": volume})
def download_nes():
    """
    modal run midi_lm/modal_config.py::stub.download_nes
    """
    from midi_lm.datasets.nes import NESDataset

    NESDataset.download("/root/data/nes")
    volume.commit()
    NESDataset.make_splits("/root/data/nes")
    volume.commit()


@stub.function(volumes={"/root/data": volume}, cpu=8)
def download_maestro():
    """
    modal run midi_lm/modal_config.py::stub.download_maestro
    """
    from midi_lm.datasets.maestro import MaestroDataset

    MaestroDataset.download("/root/data/maestro")
    volume.commit()
    MaestroDataset.make_splits("/root/data/maestro")
    volume.commit()


@stub.function(volumes={"/root/data": volume}, cpu=16, timeout=60 * 60)
def download_symphony_net():
    """
    modal run midi_lm/modal_config.py::stub.download_symphony_net
    """
    from midi_lm.datasets.symphony_net import SymphonyNetDataset

    SymphonyNetDataset.download("/root/data/symphony_net")
    volume.commit()
    SymphonyNetDataset.make_splits("/root/data/symphony_net")
    volume.commit()


@stub.function(volumes={"/root/data": volume}, cpu=16, timeout=60 * 60)
def download_giantmidi():
    """
    modal run midi_lm/modal_config.py::stub.download_giantmidi
    """
    from midi_lm.datasets.giantmidi_piano import GiantMidiPianoDataset

    GiantMidiPianoDataset.download("/root/data/giantmidi_piano")
    volume.commit()
    GiantMidiPianoDataset.make_splits("/root/data/giantmidi_piano")
    volume.commit()
