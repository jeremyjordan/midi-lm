import json
from pathlib import Path
from typing import ClassVar

import muspy
import torch

from midi_lm import logger
from midi_lm.models.multitrack_music_transformer.const import (
    DEFAULT_INSTRUMENT_MAP,
    DEFAULT_KNOWN_DURATIONS,
    DEFAULT_MAX_BEATS,
    DEFAULT_RESOLUTION,
)
from midi_lm.tokenizers import BaseTokenizer, TokenizerConfig


class MultitrackMusicTransformerTokenizer(BaseTokenizer):
    type_codes: ClassVar[dict[str, int]] = {
        "start-of-song": 0,
        "instrument": 1,
        "start-of-notes": 2,
        "note": 3,
        "end-of-song": 4,
    }

    def __init__(
        self,
        known_durations=DEFAULT_KNOWN_DURATIONS,
        max_beats=DEFAULT_MAX_BEATS,
        instrument_map=DEFAULT_INSTRUMENT_MAP,
        max_seq_len=1024,
    ):
        super().__init__()
        self.known_durations = known_durations
        self.max_beats = max_beats
        self.instrument_map = instrument_map
        self.known_instruments = list(set(v for v in instrument_map.values() if v is not None))
        self.config = TokenizerConfig(
            max_seq_len=max_seq_len,
            pad_token_id=0,
            special_token_offset=1,
        )

    def closest_known_duration(self, duration):
        # find the closest duration in the list of known durations
        return min(self.known_durations, key=lambda x: abs(x - duration))

    def encode_duration(self, duration: int):
        """Convert duration to an embedding index.
        Args:
            duration (int): Duration in ticks
        """
        # encode durations as a contiguous set of integers for the embedding lookup
        # e.g. known_durations of [1, 2, 4, 8] would be encoded as [0, 1, 2, 3]
        return self.known_durations.index(self.closest_known_duration(duration))

    def encode_instrument(self, instrument: int):
        """Convert MIDI program to an embedding index.
        Args:
            instrument (int): MIDI program number
        """
        # first map instrument into reduced set of known instruments
        mapped_instrument = self.instrument_map[instrument]
        # then encode as a contiguous set of integers for the embedding lookup
        return self.known_instruments.index(mapped_instrument)

    @property
    def vocab_size(self):
        return {
            "event_type": len(self.type_codes),
            "beat": self.max_beats + self.config.special_token_offset,
            "position": DEFAULT_RESOLUTION + self.config.special_token_offset,
            "pitch": 128 + self.config.special_token_offset,
            "duration": len(self.known_durations) + self.config.special_token_offset,
            "instrument": len(self.known_instruments) + self.config.special_token_offset,
        }

    def encode(self, music: muspy.Music):
        if music.resolution != DEFAULT_RESOLUTION:
            logger.warning(f"Warning: resolution is {music.resolution}. Adjusting to 12...")
            music = music.adjust_resolution(DEFAULT_RESOLUTION)

        resolution = music.resolution
        instruments = sorted(
            set([t.program for t in music.tracks if self.instrument_map[t.program] is not None])
        )

        # results format: (type, beat, position, pitch, duration, instrument)
        result = []

        # start of song
        result.append(
            (
                self.type_codes["start-of-song"],
                self.config.pad_token_id,
                self.config.pad_token_id,
                self.config.pad_token_id,
                self.config.pad_token_id,
                self.config.pad_token_id,
            )
        )

        # instruments
        for instrument in instruments:
            result.append(
                (
                    self.type_codes["instrument"],
                    self.config.pad_token_id,
                    self.config.pad_token_id,
                    self.config.pad_token_id,
                    self.config.pad_token_id,
                    self.encode_instrument(instrument) + self.config.special_token_offset,
                )
            )

        # start of notes
        result.append(
            (
                self.type_codes["start-of-notes"],
                self.config.pad_token_id,
                self.config.pad_token_id,
                self.config.pad_token_id,
                self.config.pad_token_id,
                self.config.pad_token_id,
            )
        )

        # notes
        for track in music.tracks:
            if track.program not in instruments:
                continue
            instrument_idx = self.encode_instrument(track.program)
            for note in track.notes:
                beat, position = divmod(note.time, resolution)
                if beat >= self.max_beats:
                    break
                result.append(
                    (
                        self.type_codes["note"],
                        beat + self.config.special_token_offset,
                        position + self.config.special_token_offset,
                        note.pitch + self.config.special_token_offset,
                        self.encode_duration(note.duration) + self.config.special_token_offset,
                        instrument_idx + self.config.special_token_offset,
                    )
                )

        # end of song
        result.append(
            (
                self.type_codes["end-of-song"],
                self.config.pad_token_id,
                self.config.pad_token_id,
                self.config.pad_token_id,
                self.config.pad_token_id,
                self.config.pad_token_id,
            )
        )

        # sort notes by type, beat, position, then instrument
        result.sort(key=lambda x: (x[0], x[1], x[2], x[5]))
        codes = torch.tensor(result, dtype=torch.long)

        # check if the sequence is too long
        if codes.shape[0] > self.config.max_seq_len:
            logger.debug(
                f"Sequence is too long ({codes.shape[0]}), truncating to {self.config.max_seq_len}"
            )
            # preserve the end of sequence token
            codes = torch.concat([codes[: self.config.max_seq_len - 1], codes[-1:]], dim=0)

        return {
            "event_type": codes[:, 0],
            "beat": codes[:, 1],
            "position": codes[:, 2],
            "pitch": codes[:, 3],
            "duration": codes[:, 4],
            "instrument": codes[:, 5],
        }

    def decode(self, tokens, resolution=DEFAULT_RESOLUTION):
        # NOTE: it's important that the resolution matches what the model was trained on
        # TODO: should I make the tempo adjustable?
        music = muspy.Music(resolution=resolution, tempos=[muspy.Tempo(time=0, qpm=100)])

        tracks: dict[int, muspy.Track] = {}  # {instrument: track}

        for event_type, beat, position, pitch, duration, instrument in zip(
            tokens["event_type"],
            tokens["beat"],
            tokens["position"],
            tokens["pitch"],
            tokens["duration"],
            tokens["instrument"],
            strict=True,
        ):
            beat = int(beat) - self.config.special_token_offset
            position = int(position) - self.config.special_token_offset
            pitch = int(pitch) - self.config.special_token_offset
            duration = self.known_durations[int(duration) - self.config.special_token_offset]
            instrument = self.known_instruments[int(instrument) - self.config.special_token_offset]

            if event_type in [
                self.type_codes["start-of-song"],
                self.type_codes["start-of-notes"],
            ]:
                continue
            if event_type == self.type_codes["end-of-song"]:
                break
            if event_type == self.type_codes["instrument"]:
                tracks[instrument] = muspy.Track(program=instrument)
            elif event_type == self.type_codes["note"]:
                time = beat * resolution + position
                tracks[instrument].notes.append(muspy.Note(time=time, pitch=pitch, duration=duration))
            else:
                raise ValueError(f"Unknown event type: {event_type}")

        music.tracks = list(tracks.values())
        return music

    def save(self, filepath: str | Path):
        """Output a JSON file of the tokenizer state.

        Args:
            filepath (str | Path): Path for the JSON output file.
        """
        filepath = Path(filepath)
        body = {
            "known_durations": self.known_durations,
            "max_beats": self.max_beats,
            "max_seq_len": self.config.max_seq_len,
        }
        filepath.write_text(json.dumps(body))

    @classmethod
    def load(cls, filepath: str | Path):
        """Load a tokenizer from a JSON file.

        Args:
            filepath (str | Path): Path to the JSON file containing tokenizer state.

        Returns:
            MultitrackMusicTransformerTokenizer: The loaded tokenizer.
        """
        filepath = Path(filepath)
        body = json.loads(filepath.read_text())
        return cls(**body)

    def __eq__(self, __value: object) -> bool:
        return (
            self.config == getattr(__value, "config", None)
            and self.type_codes == getattr(__value, "type_codes", None)
            and self.known_durations == getattr(__value, "known_durations", None)
            and self.max_beats == getattr(__value, "max_beats", None)
            and self.instrument_map == getattr(__value, "instrument_map", None)
        )
