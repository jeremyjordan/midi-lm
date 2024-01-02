import muspy

from midi_lm.transforms import Compose
from midi_lm.transforms.crop import RandomCrop


def test_simple_compose(example_piano_track: muspy.Music):
    music = example_piano_track
    transform = Compose([RandomCrop(n_beats=10)])
    result = transform(music)

    assert isinstance(result, muspy.Music)
    assert len(result.tracks[0]) < len(music.tracks[0])
    assert result.get_end_time() <= 10 * result.resolution
