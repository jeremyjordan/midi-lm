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
CUSTOM_STYLE = """
midi-player {
  display: block;
  width: inherit;
  margin: 4px;
  margin-bottom: 0;
}
midi-player::part(control-panel) {
  background: #f2f5f6;
  border: 2px solid #000;
  border-radius: 10px 10px 0 0;
}
midi-player::part(play-button) {
  color: #333333;
  border: 2px solid currentColor;
  background-color: #D8D8D8;
  border-radius: 20px;
  transition: all 0.2s;
  content: 'hello';
}
midi-player::part(play-button):hover {
  color: #333333;
  background-color: #FFFFFF;
  border-radius: 10px;
}
midi-player::part(time) {
  font-family: monospace;
}

/* Custom visualizer style */
midi-visualizer .piano-roll-visualizer {
  background: #FFFFFF;
  border: 2px solid black;
  border-top: none;
  border-radius: 0 0 10px 10px;
  margin: 4px;
  margin-top: 0;
  overflow: auto;
}
midi-visualizer svg rect.note {
  opacity: 0.6;
  stroke-width: 2;
}
midi-visualizer svg rect.note[data-instrument="0"]{
  fill: #A5A5A5;
  stroke: #333333;
}
midi-visualizer svg rect.note[data-instrument="2"]{
  fill: #A5A5A5;
  stroke: #333333;
}
midi-visualizer svg rect.note.active {
  opacity: 0.9;
  stroke: #000;
  fill: #99C3FF;
}
"""


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
    <style>
    {CUSTOM_STYLE}
    </style>
    <script src="{MAGENTA_JS}"></script>
    <p>{title}</p>
    <section>
      <midi-player src="{data_url}" sound-font="" visualizer="#midi-visualizer">
      </midi-player>
      <midi-visualizer src="{data_url}" type="{vis_type}" id="midi-visualizer">
      </midi-visualizer>
    </section>
    """

    output = f"""
    <iframe srcdoc="{escape(rendered_html)}" width="{width}" height="{height}"
        style="border:none !important;"
        allowfullscreen webkitallowfullscreen mozallowfullscreen>
    </iframe>
    """
    return output
