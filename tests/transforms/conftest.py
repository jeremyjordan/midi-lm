import muspy
import pytest


@pytest.fixture()
def example_piano_track() -> muspy.Music:
    piano_track = muspy.Track(
        program=0,
        name="A major scale",
        notes=[
            muspy.Note(time=0, pitch=69, duration=2, velocity=80),
            muspy.Note(time=2, pitch=71, duration=1, velocity=80),
            muspy.Note(time=3, pitch=73, duration=1, velocity=80),
            muspy.Note(time=4, pitch=74, duration=1, velocity=80),
            muspy.Note(time=5, pitch=76, duration=1, velocity=80),
            muspy.Note(time=6, pitch=78, duration=1, velocity=80),
            muspy.Note(time=7, pitch=80, duration=1, velocity=80),
            muspy.Note(time=8, pitch=81, duration=2, velocity=80),
            muspy.Note(time=10, pitch=80, duration=1, velocity=80),
            muspy.Note(time=11, pitch=78, duration=1, velocity=80),
            muspy.Note(time=12, pitch=76, duration=1, velocity=80),
            muspy.Note(time=13, pitch=74, duration=1, velocity=80),
            muspy.Note(time=14, pitch=73, duration=1, velocity=80),
            muspy.Note(time=15, pitch=71, duration=1, velocity=80),
            muspy.Note(time=16, pitch=69, duration=2, velocity=80),
            muspy.Note(time=20, pitch=69, duration=2, velocity=80),
            muspy.Note(time=22, pitch=73, duration=2, velocity=80),
            muspy.Note(time=24, pitch=76, duration=2, velocity=80),
            muspy.Note(time=26, pitch=81, duration=2, velocity=80),
            muspy.Note(time=28, pitch=76, duration=2, velocity=80),
            muspy.Note(time=30, pitch=73, duration=2, velocity=80),
            muspy.Note(time=32, pitch=69, duration=2, velocity=80),
        ],
    )
    music = muspy.Music(resolution=2, tracks=[piano_track])
    return music
