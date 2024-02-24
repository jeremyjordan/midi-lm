"""
Script to iterate over a dataloader in order to profile it.

pip install py-spy

sudo py-spy record -o outputs/profile.json --subprocesses --format speedscope -- python \
    midi_lm/datasets/scripts/profile_dataloader.py \
    dataset=nes tokenizer=mmt transforms=crop-transpose dataset.num_workers=8
"""

import time

import hydra

from midi_lm import logger
from midi_lm.config import TrainingConfig
from midi_lm.config.transforms import create_transforms
from midi_lm.datasets.base import MusicDataModule
from midi_lm.tokenizers import BaseTokenizer

PROFILE_STEPS = 100


# CLI entrypoint
@hydra.main(version_base=None, config_name="config")
def main(config: TrainingConfig) -> None:
    logger.info("Preparing dataset...")
    tokenizer: BaseTokenizer = hydra.utils.instantiate(config.tokenizer)
    collate_fn = hydra.utils.get_method(config.collator._target_)
    transforms = create_transforms(config.transforms)
    dataset: MusicDataModule = hydra.utils.instantiate(
        config.dataset,
        tokenizer=tokenizer,
        collate_fn=collate_fn,
        transforms=transforms,
    )
    dataset.setup("fit")

    train_dataloader = dataset.train_dataloader()

    logger.info(f"Profiling dataloader with {PROFILE_STEPS} steps")

    start = time.time()
    steps = 0
    while steps < PROFILE_STEPS:
        for _ in train_dataloader:
            steps += 1
            if steps >= PROFILE_STEPS:
                break
    end = time.time()
    logger.info(f"Finished profiling {PROFILE_STEPS} steps in {end - start:.2f} seconds")


if __name__ == "__main__":
    main()
