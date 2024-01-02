"""
Created volume from CLI:

    modal volume create datasets-vol

Upload local datasets:

    modal volume put datasets-vol data/ 
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
