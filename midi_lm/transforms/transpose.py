import random

import muspy


class TransposeNotes:
    def __init__(self, min_semitones: int = -5, max_semitones: int = 5):
        self.min_semitones = min_semitones
        self.max_semitones = max_semitones

    def __call__(self, music: muspy.Music):
        semitone = random.randint(self.min_semitones, self.max_semitones)
        music = music.transpose(semitone)
        return music

    def __eq__(self, __value: object) -> bool:
        return all(
            [
                self.min_semitones == getattr(__value, "min_semitones", None),
                self.max_semitones == getattr(__value, "max_semitones", None),
            ]
        )
