"""
Common commands for volumes:

    modal volume create datasets-vol
    modal volume put datasets-vol data/ 
    modal volume ls datasets-vol
    modal volume rm datasets-vol -r scales/
"""

import modal

from midi_lm import ROOT_DIR

requirements_file = ROOT_DIR.parent / "requirements.txt"

remote_image = (
    modal.Image.debian_slim(python_version="3.11.2")
    .apt_install("git")
    .pip_install_from_requirements(requirements_file.as_posix())
    .pip_install("gdown")
)

stub = modal.Stub(
    name="midi-language-modeling",
    image=remote_image,
    secrets=[
        modal.Secret.from_name("wandb-secret"),
    ],
)

volume = modal.Volume.persisted("datasets-vol")


# remote dataset download functions
@stub.function(volumes={"/root/data": volume})
def download_scales():
    """
    modal run midi_lm/modal_config.py::stub.download_scales
    """
    from midi_lm.datasets.scales import ScalesDataset

    ScalesDataset.download("/root/data/scales")
    ScalesDataset.make_splits("/root/data/scales")
    volume.commit()


@stub.function(volumes={"/root/data": volume})
def download_bach():
    """
    modal run midi_lm/modal_config.py::stub.download_bach
    """
    from midi_lm.datasets.bach_chorales import BachChoralesDataset

    BachChoralesDataset.download("/root/data/bach_chorales")
    BachChoralesDataset.make_splits("/root/data/bach_chorales")
    volume.commit()


@stub.function(volumes={"/root/data": volume})
def download_nes():
    """
    modal run midi_lm/modal_config.py::stub.download_nes
    """
    from midi_lm.datasets.nes import NESDataset

    NESDataset.download("/root/data/nes")
    NESDataset.make_splits("/root/data/nes")
    volume.commit()


@stub.function(volumes={"/root/data": volume}, cpu=8)
def download_maestro():
    """
    modal run midi_lm/modal_config.py::stub.download_maestro
    """
    from midi_lm.datasets.maestro import MaestroDataset

    MaestroDataset.download("/root/data/maestro")
    MaestroDataset.make_splits("/root/data/maestro")
    volume.commit()


@stub.function(volumes={"/root/data": volume}, cpu=16, timeout=60 * 60)
def download_symphony_net():
    """
    modal run midi_lm/modal_config.py::stub.download_symphony_net
    """
    from midi_lm.datasets.symphony_net import SymphonyNetDataset

    SymphonyNetDataset.download("/root/data/symphony_net")
    SymphonyNetDataset.make_splits("/root/data/symphony_net")
    volume.commit()
