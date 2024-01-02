"""
A simple tokenizer that represents a song as a series of (time_shift, pitch,
duration) tuples.

This is similar to the Time Shift Duration tokenizer, but instead of predicting
values in a sequence we predict them jointly.

https://miditok.readthedocs.io/en/v2.1.7/tokenizations.html#tsd
"""

import json
from collections import Counter
from pathlib import Path

import muspy
import torch

from midi_lm import logger
from midi_lm.tokenizers import BaseTokenizer, TokenizerConfig

DEFAULT_RESOLUTION = 12
DEFAULT_KNOWN_TIME_SHIFTS = [0, 6, 12, 24, 32, 48, 64, 96, 128]
DEFAULT_KNOWN_DURATIONS = [2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]


class TimeShiftPitchDurationTokenizer(BaseTokenizer):
    def __init__(
        self,
        known_time_shifts: list[int] = DEFAULT_KNOWN_TIME_SHIFTS,
        known_durations: list[int] = DEFAULT_KNOWN_DURATIONS,
        max_seq_len=1024,
    ) -> None:
        self.known_time_shifts = known_time_shifts
        self.known_durations = known_durations
        self.config = TokenizerConfig(
            max_seq_len=max_seq_len, pad_token_id=0, bos_token_id=1, eos_token_id=2, special_token_offset=3
        )

    @property
    def vocab_size(self):
        return {
            "time_shift": len(self.known_time_shifts) + 1,  # padding token
            "pitch": 128 + self.config.special_token_offset,
            "duration": len(self.known_durations) + 1,  # padding token
        }

    def encode(self, music: muspy.Music) -> dict[str, torch.Tensor]:
        assert len(music.tracks) == 1, "This tokenizer only supports single track music"

        music = music.sort()
        result = []
        last_time = 0
        for note in music.tracks[0].notes:
            time_shift = note.time - last_time
            last_time = note.time
            result.append(
                (
                    time_shift,
                    note.pitch,
                    note.duration,
                )
            )
        codes = torch.tensor(result, dtype=torch.long)

        # check if the sequence is too long
        if codes.shape[0] > self.config.max_seq_len - 2:
            logger.debug(
                f"Sequence is too long ({codes.shape[0]}), truncating to {self.config.max_seq_len}"
            )
            # preserve the end of sequence token
            codes = codes[: self.config.max_seq_len - 2]  # save space for start and end tokens

        # map to the closest known time shift
        known_time_shifts = torch.tensor(self.known_time_shifts, dtype=torch.long)
        time_shifts = codes[:, 0]
        distances = torch.abs(time_shifts[:, None] - known_time_shifts[None, :])
        codes[:, 0] = torch.argmin(distances, dim=1)

        # map to closest known duration
        known_durations = torch.tensor(self.known_durations, dtype=torch.long)
        durations = codes[:, 2]
        distances = torch.abs(durations[:, None] - known_durations[None, :])
        codes[:, 2] = torch.argmin(distances, dim=1)

        # add start and end tokens
        codes[:, 0] += 1
        codes[:, 1] += self.config.special_token_offset
        codes[:, 2] += 1
        codes = torch.cat(
            [
                torch.tensor(
                    [
                        [
                            self.config.pad_token_id,
                            self.config.bos_token_id,
                            self.config.pad_token_id,
                        ],
                    ]
                ),
                codes,
                torch.tensor(
                    [
                        [
                            self.config.pad_token_id,
                            self.config.eos_token_id,
                            self.config.pad_token_id,
                        ],
                    ]
                ),
            ],
            dim=0,
        )

        return {
            "time_shift": codes[:, 0],
            "pitch": codes[:, 1],
            "duration": codes[:, 2],
        }

    def decode(self, tokens: dict[str, torch.Tensor], resolution=DEFAULT_RESOLUTION) -> muspy.Music:
        music = muspy.Music(resolution=resolution, tempos=[muspy.Tempo(time=0, qpm=100)])
        track = muspy.Track(program=0)

        current_time = 0
        for time_shift, pitch, duration in zip(
            tokens["time_shift"], tokens["pitch"], tokens["duration"], strict=True
        ):
            if pitch == self.config.bos_token_id:
                continue
            if pitch == self.config.eos_token_id:
                break
            time_shift = self.known_time_shifts[time_shift - 1]
            current_time += time_shift
            duration = self.known_durations[duration - 1]
            track.notes.append(
                muspy.Note(
                    time=current_time,
                    pitch=int(pitch) - self.config.special_token_offset,
                    duration=duration,
                    velocity=64,
                )
            )
        music.tracks.append(track)
        return music

    def save(self, filepath):
        filepath = Path(filepath)
        body = {
            "known_durations": self.known_durations,
            "known_time_shifts": self.known_time_shifts,
            "max_seq_len": self.config.max_seq_len,
        }
        filepath.write_text(json.dumps(body))

    @classmethod
    def load(cls, filepath):
        filepath = Path(filepath)
        body = json.loads(filepath.read_text())
        return cls(**body)

    @classmethod
    def fit(cls, midi_files, resolution=DEFAULT_RESOLUTION):
        timeshift_counter = Counter()
        pitch_counter = Counter()
        duration_counter = Counter()

        for f in midi_files:
            music = muspy.load_json(f)
            music = music.adjust_resolution(resolution)
            for track in music.tracks:
                last_time = 0
                for note in track.notes:
                    time_shift = note.time - last_time

                    timeshift_counter[time_shift] += 1
                    pitch_counter[note.pitch] += 1
                    duration_counter[note.duration] += 1

                    last_time = note.time

        return {
            "time_shift": timeshift_counter,
            "pitch": pitch_counter,
            "duration": duration_counter,
        }

    def __eq__(self, __value: object) -> bool:
        return (
            self.config == getattr(__value, "config", None)
            and self.known_time_shifts == getattr(__value, "known_time_shifts", None)
            and self.known_durations == getattr(__value, "known_durations", None)
        )


if __name__ == "__main__":
    p = Path("/Users/jeremyjordan/Documents/Code/midi-lm/data/bach_chorales/JSB Chorales/splits.json")
    splits = json.loads(p.read_text())
    train_files = [
        Path("/Users/jeremyjordan/Documents/Code/midi-lm/data/bach_chorales/JSB Chorales/", f).as_posix()
        for f in splits["train"]
    ]

    results = TimeShiftPitchDurationTokenizer.fit(train_files)
    print(results)
