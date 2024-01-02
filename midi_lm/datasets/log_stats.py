import json
from collections import defaultdict
from pathlib import Path

import hydra
import muspy
import numpy as np
import wandb
from tqdm import tqdm

from midi_lm import logger
from midi_lm.config import TrainingConfig
from midi_lm.config.transforms import create_transforms
from midi_lm.datasets.base import MusicDataModule, MusicDataset
from midi_lm.tokenizers import BaseTokenizer


def collect_stats_from_dataset(dataset: MusicDataset):
    files = dataset.midi_files

    results = defaultdict(list)
    for f in tqdm(files):
        try:
            music = muspy.load_json(f)

            # compute metrics
            n_beats = music.get_end_time() // music.resolution
            n_tracks = len(music.tracks)
            n_notes = sum(len(track.notes) for track in music.tracks)
            n_instruments = len(set(track.program for track in music.tracks))
            pitch_range = muspy.pitch_range(music)
            pitch_entropy = muspy.pitch_entropy(music)
            scale_consistency = muspy.scale_consistency(music)
            empty_beat_ratio = muspy.empty_beat_rate(music)
            avg_concurrent_pitches = muspy.polyphony(music)

            # add to results
            results["file"].append(f.name)
            results["n_beats"].append(n_beats)
            results["n_tracks"].append(n_tracks)
            results["n_notes"].append(n_notes)
            results["n_instruments"].append(n_instruments)
            results["pitch_range"].append(pitch_range)
            results["pitch_entropy"].append(pitch_entropy)
            results["scale_consistency"].append(scale_consistency)
            results["empty_beat_ratio"].append(empty_beat_ratio)
            results["avg_concurrent_pitches"].append(avg_concurrent_pitches)
        except Exception as e:
            logger.warning(f"Error processing {f}: {e}")

    return results


def _filter_nan(values):
    return [v for v in values if not np.isnan(v)]


def log_to_wandb(stats, prefix):
    logger.info(f"Logging {prefix} stats to wandb...")
    wandb.log(
        {
            f"{prefix}/n_files": len(stats["file"]),
            f"{prefix}/avg_beats": np.mean(stats["n_beats"]),
            f"{prefix}/avg_tracks": np.mean(stats["n_tracks"]),
            f"{prefix}/avg_notes": np.mean(stats["n_notes"]),
            f"{prefix}/avg_instruments": np.mean(stats["n_instruments"]),
            f"{prefix}/total_beats": np.sum(stats["n_beats"]),
            f"{prefix}/total_tracks": np.sum(stats["n_tracks"]),
            f"{prefix}/total_notes": np.sum(stats["n_notes"]),
            f"{prefix}/n_beats": wandb.Histogram(stats["n_beats"]),
            f"{prefix}/n_tracks": wandb.Histogram(stats["n_tracks"]),
            f"{prefix}/n_notes": wandb.Histogram(stats["n_notes"]),
            f"{prefix}/n_instruments": wandb.Histogram(stats["n_instruments"]),
            f"{prefix}/pitch_range": wandb.Histogram(stats["pitch_range"]),
            f"{prefix}/pitch_entropy": wandb.Histogram(_filter_nan(stats["pitch_entropy"])),
            f"{prefix}/scale_consistency": wandb.Histogram(_filter_nan(stats["scale_consistency"])),
            f"{prefix}/empty_beat_ratio": wandb.Histogram(_filter_nan(stats["empty_beat_ratio"])),
            f"{prefix}/avg_concurrent_pitches": wandb.Histogram(
                _filter_nan(stats["avg_concurrent_pitches"])
            ),
        }
    )

    # save raw file
    assert wandb.run is not None
    filepath = Path(wandb.run.dir, f"{prefix}_stats.json")
    filepath.write_text(json.dumps(stats))
    wandb.save(filepath.as_posix())


# CLI entrypoint
@hydra.main(version_base=None, config_name="config")
def main(config: TrainingConfig) -> None:
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

    wandb.init(
        project="music-dataset-stats",
        dir="wandb",
    )
    wandb.config.update({"dataset": dataset.__class__.__name__})

    train_stats = collect_stats_from_dataset(dataset.train_dataset)
    log_to_wandb(train_stats, "train")

    val_stats = collect_stats_from_dataset(dataset.val_dataset)
    log_to_wandb(val_stats, "val")


if __name__ == "__main__":
    main()
