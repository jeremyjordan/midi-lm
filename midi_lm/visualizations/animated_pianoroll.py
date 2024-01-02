"""
This is heavily inspired by Dr. Scott Hawley's library https://github.com/drscotthawley/midi-player

Here, we take a muspy.Music object and directly convert it to a visual representation of a piano roll.
"""

import base64
import io
from html import escape
from typing import Literal

import muspy

MAGENTA_JS_REQUIREMENTS = [
    "npm/tone@14.7.58",
    "npm/@magenta/music@1.23.1/es6/core.js",
    "npm/focus-visible@5",
    "npm/html-midi-player@1.5.0",
]
MAGENTA_JS = f"https://cdn.jsdelivr.net/combine/{','.join(MAGENTA_JS_REQUIREMENTS)}"
VISUALIZATION_OPTIONS = Literal["piano-roll", "waterfall", "staff"]


def midi_player_iframe(
    music: muspy.Music, vis_type: VISUALIZATION_OPTIONS = "piano-roll", width="100%", height="400", title=""
) -> str:
    """Generate an iframe for rendering an animated MIDI player.

    Args:
        music (muspy.Music): the MIDI music to render
        vis_type (str, optional): the type of visualizer to use, defaults to "piano-roll"
        width (str, optional): width of the iframe, defaults to "100%"
        height (str, optional): height of the iframe, defaults to "400"

    Returns:
        str: an HTML string containing the iframe
    """

    # Convert MusPy Music object to MIDI and encode as base64
    midi_bytes_io = io.BytesIO()
    muspy.to_pretty_midi(music).write(midi_bytes_io)
    midi_bytes = midi_bytes_io.getvalue()
    data_url = "data:audio/midi;base64," + base64.b64encode(midi_bytes).decode("utf-8")

    # Create the HTML snippet
    rendered_html = f"""
    <script src="{MAGENTA_JS}"></script>
    <p>{title}</p>
    <midi-player src="{data_url}" sound-font visualizer="#myVisualizer"></midi-player>
    <midi-visualizer type="{vis_type}" id="myVisualizer" style="background: #fff;"></midi-visualizer>
    """

    output = f"""
    <iframe srcdoc="{escape(rendered_html)}" width="{width}" height="{height}"
        style="border:none !important;"
        allowfullscreen webkitallowfullscreen mozallowfullscreen>
    </iframe>
    """
    return output
