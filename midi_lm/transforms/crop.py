import random

import muspy

from midi_lm import logger


def crop_list(values: list, start_time: int, end_time: int) -> list:
    """Crop the list of values to the given start and end time."""
    return [x for x in values if start_time <= x.time < end_time]


def crop_music(music: muspy.Music, start_beat: int, end_beat: int) -> muspy.Music:
    assert start_beat >= 0, "start_beat must be greater than or equal to 0"
    assert end_beat > 0, "end_beat must be greater than 0"
    assert start_beat < end_beat, "start_beat must be less than end_beat"

    # Make a deepcopy of the music so we don't modify the original
    # TODO this is a bit inefficient and not necessary as long as we're always loading from file
    music = music.deepcopy()
    start_time = start_beat * music.resolution
    end_time = end_beat * music.resolution

    music.tempos = crop_list(music.tempos, start_time, end_time)
    music.key_signatures = crop_list(music.key_signatures, start_time, end_time)
    music.time_signatures = crop_list(music.time_signatures, start_time, end_time)
    music.beats = crop_list(music.beats, start_time, end_time)
    music.lyrics = crop_list(music.lyrics, start_time, end_time)
    music.annotations = crop_list(music.annotations, start_time, end_time)

    new_tracks = []
    for track in music.tracks:
        notes = crop_list(track.notes, start_time, end_time)
        chords = crop_list(track.chords, start_time, end_time)
        lyrics = crop_list(track.lyrics, start_time, end_time)
        annotations = crop_list(track.annotations, start_time, end_time)
        new_tracks.append(
            muspy.Track(
                program=track.program,
                is_drum=track.is_drum,
                name=track.name,
                notes=notes,
                chords=chords,
                lyrics=lyrics,
                annotations=annotations,
            )
        )
    music.tracks = new_tracks
    music.adjust_time(lambda t: t - start_time)
    return music


class RandomCrop:
    def __init__(self, n_beats: int = 16) -> None:
        assert n_beats > 0, "n_beats must be greater than 0"
        self.n_beats = n_beats

    def __call__(self, music: muspy.Music) -> muspy.Music:
        total_beats = music.get_end_time() // music.resolution
        if total_beats <= self.n_beats:
            logger.debug("Music is too short to crop")
            return music
        start_beat = random.randint(0, total_beats - self.n_beats)
        end_beat = start_beat + self.n_beats
        logger.debug(f"Randomly cropping music from beat {start_beat} to {end_beat}")
        logger.debug(f"Total beats: {total_beats}")
        return crop_music(music, start_beat, end_beat)

    def __eq__(self, __value: object) -> bool:
        return self.n_beats == getattr(__value, "n_beats", None)
