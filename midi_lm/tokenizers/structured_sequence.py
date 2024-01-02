import json
from pathlib import Path

import muspy
import torch

from midi_lm.tokenizers import BaseTokenizer, TokenizerConfig

DEFAULT_RESOLUTION = 12


def crop_sequence(tensor, bos_idx: int | None = None, eos_idx: int | None = None):
    assert len(tensor.shape) == 1, "expected 1 dimensional tensor"
    has_bos = (tensor == bos_idx).nonzero(as_tuple=True)[0] if bos_idx is not None else []
    has_eos = (tensor == eos_idx).nonzero(as_tuple=True)[0] if eos_idx is not None else []
    start = has_bos[0].item() + 1 if len(has_bos) > 0 else 0
    end = has_eos[0].item() if len(has_eos) > 0 else len(tensor)
    return tensor[start:end]


class StructuredSequenceTokenizer(BaseTokenizer):
    def __init__(self, max_seq_len=1024):
        super().__init__()
        self.config = TokenizerConfig(
            max_seq_len=max_seq_len,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            special_token_offset=3,
        )

        # ranges are from [start, end)
        self.pitches = torch.arange(0, 128)
        self.velocities = torch.linspace(1, 127, 32, dtype=torch.long)
        self.durations = torch.arange(1, 257)
        self.time_shifts = torch.arange(0, 257)

        self.pitch_offset = self.config.special_token_offset
        self.velocity_offset = self.pitch_offset + len(self.pitches)
        self.duration_offset = self.velocity_offset + len(self.velocities)
        self.time_shift_offset = self.duration_offset + len(self.durations)

    @property
    def vocab_size(self):
        return (
            self.config.special_token_offset
            + len(self.pitches)
            + len(self.velocities)
            + len(self.durations)
            + len(self.time_shifts)
        )

    @property
    def token_types(self):
        return {
            "special": (0, self.config.special_token_offset - 1),
            "pitch": (self.pitch_offset, self.pitch_offset + len(self.pitches) - 1),
            "velocity": (
                self.velocity_offset,
                self.velocity_offset + len(self.velocities) - 1,
            ),
            "duration": (
                self.duration_offset,
                self.duration_offset + len(self.durations) - 1,
            ),
            "timeshift": (
                self.time_shift_offset,
                self.time_shift_offset + len(self.time_shifts) - 1,
            ),
        }

    def encode(self, music: muspy.Music) -> dict[str, torch.Tensor]:
        assert len(music.tracks) == 1, "This tokenizer currently only supports single track music"
        music = music.sort()
        tokens = [self.config.bos_token_id]

        last_time = 0
        for note in music.tracks[0].notes:
            time_shift = note.time - last_time
            last_time = note.time
            velocity_bin = int(torch.argmin(torch.abs(self.velocities - note.velocity)).item())
            duration_bin = int(torch.argmin(torch.abs(self.durations - note.duration)).item())
            time_shift_bin = int(torch.argmin(torch.abs(self.time_shifts - time_shift)).item())
            tokens.append(note.pitch + self.pitch_offset)
            tokens.append(velocity_bin + self.velocity_offset)
            tokens.append(duration_bin + self.duration_offset)
            tokens.append(time_shift_bin + self.time_shift_offset)

        tokens.append(self.config.eos_token_id)

        return {"token_ids": torch.tensor(tokens, dtype=torch.long)}

    def decode(self, tokens: dict[str, torch.Tensor], resolution=DEFAULT_RESOLUTION) -> muspy.Music:
        music = muspy.Music(resolution=resolution, tempos=[muspy.Tempo(time=0, qpm=100)])
        track = muspy.Track(program=0)

        token_tensor = tokens["token_ids"]

        # if a BOS is detected, drop all tokens before it
        # if an EOS is detected, drop all tokens after it
        token_tensor = crop_sequence(
            token_tensor,
            bos_idx=self.config.bos_token_id,
            eos_idx=self.config.eos_token_id,
        )

        # the remaining sequence should be a multiple of 4
        # we'll ignore any remaining tokens
        seq_len = token_tensor.shape[0]
        remainder = seq_len % 4
        if remainder > 0:
            token_tensor = token_tensor[:-remainder]
        assert token_tensor.shape[0] % 4 == 0, "expected a multiple of 4 (structured) tokens"

        # split the sequence into groups of 4 by reshaping the 1 dimensional tensor
        # convert the token ids back to their original values
        token_tensor = token_tensor.reshape(-1, 4).to("cpu")
        pitches = self.pitches[token_tensor[:, 0] - self.pitch_offset]
        velocities = self.velocities[token_tensor[:, 1] - self.velocity_offset]
        durations = self.durations[token_tensor[:, 2] - self.duration_offset]
        time_shifts = self.time_shifts[token_tensor[:, 3] - self.time_shift_offset]

        # convert the tokens into notes
        current_time = 0
        for pitch, velocity, duration, timeshift in zip(
            pitches, velocities, durations, time_shifts, strict=True
        ):
            current_time += int(timeshift)
            track.notes.append(
                muspy.Note(
                    time=current_time,
                    pitch=int(pitch),
                    duration=int(duration),
                    velocity=int(velocity),
                )
            )

        music.tracks.append(track)
        return music

    def save(self, filepath):
        filepath = Path(filepath)
        body = {
            "max_seq_len": self.config.max_seq_len,
        }
        filepath.write_text(json.dumps(body))

    @classmethod
    def load(cls, filepath):
        filepath = Path(filepath)
        body = json.loads(filepath.read_text())
        return cls(**body)

    def __eq__(self, __value: object) -> bool:
        return (
            self.config == getattr(__value, "config", None)
            and self.pitches == getattr(__value, "pitches", None)
            and self.velocities == getattr(__value, "velocities", None)
            and self.durations == getattr(__value, "durations", None)
            and self.time_shifts == getattr(__value, "time_shifts", None)
        )
