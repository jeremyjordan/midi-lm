import muspy


def compute_song_metrics(music: muspy.Music) -> dict[str, float]:
    """Compute metrics for a single song"""
    n_beats = music.get_end_time() // music.resolution
    n_tracks = len(music.tracks)
    n_notes = sum(len(track.notes) for track in music.tracks)
    n_instruments = len(set(track.program for track in music.tracks))
    pitch_range = muspy.pitch_range(music)
    pitch_entropy = muspy.pitch_entropy(music)
    scale_consistency = muspy.scale_consistency(music)
    empty_beat_ratio = muspy.empty_beat_rate(music)
    avg_concurrent_pitches = muspy.polyphony(music)
    return {
        "n_beats": n_beats,
        "n_tracks": n_tracks,
        "n_notes": n_notes,
        "n_instruments": n_instruments,
        "pitch_range": pitch_range,
        "pitch_entropy": pitch_entropy,
        "scale_consistency": scale_consistency,
        "empty_beat_ratio": empty_beat_ratio,
        "avg_concurrent_pitches": avg_concurrent_pitches,
    }
