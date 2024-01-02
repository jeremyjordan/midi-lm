DEFAULT_MAX_BEATS = 256
DEFAULT_RESOLUTION = 12
DEFAULT_KNOWN_DURATIONS = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    15,
    16,
    18,
    20,
    21,
    24,
    30,
    36,
    40,
    42,
    48,
    60,
    72,
    84,
    96,
    120,
    144,
    168,
    192,
    384,
]


DEFAULT_INSTRUMENT_MAP = {
    # ---------- Piano ----------
    0: 0,  # Acoustic Grand Piano -> Acoustic Grand Piano
    1: 0,  # Bright Acoustic Piano -> Acoustic Grand Piano
    2: 0,  # Electric Grand Piano -> Acoustic Grand Piano
    3: 0,  # Honky-tonk Piano -> Acoustic Grand Piano
    4: 4,  # Electric Piano 1 -> Electric Piano 1
    5: 4,  # Electric Piano 2 -> Electric Piano 1
    6: 6,  # Harpsichord -> Harpsichord
    7: 7,  # Clavinet -> Clavinet
    # ---------- Chromatic Percussion ----------
    8: 8,  # Celesta -> Celesta
    9: 9,  # Glockenspiel -> Glockenspiel
    10: 10,  # Music Box -> Music Box
    11: 11,  # Vibraphone -> Vibraphone
    12: 12,  # Marimba -> Marimba
    13: 13,  # Xylophone -> Xylophone
    14: 14,  # Tubular Bells -> Tubular Bells
    15: 15,  # Dulcimer -> Dulcimer
    # ---------- Organ ----------
    16: 16,  # Drawbar Organ -> Drawbar Organ
    17: 16,  # Percussive Organ -> Drawbar Organ
    18: 16,  # Rock Organ -> Drawbar Organ
    19: 19,  # Church Organ -> Church Organ
    20: 16,  # Reed Organ -> Drawbar Organ
    21: 21,  # Accordion -> Accordion
    22: 22,  # Harmonica -> Harmonica
    23: 23,  # Tango Accordion -> Tango Accordion
    # ---------- Guitar ----------
    24: 24,  # Acoustic Guitar (nylon) -> Acoustic Guitar (nylon)
    25: 25,  # Acoustic Guitar (steel) -> Acoustic Guitar (steel)
    26: 26,  # Electric Guitar (jazz) -> Electric Guitar (jazz)
    27: 26,  # Electric Guitar (clean) -> Electric Guitar (jazz)
    28: 26,  # Electric Guitar (muted) -> Electric Guitar (jazz)
    29: 26,  # Overdriven Guitar -> Electric Guitar (jazz)
    30: 26,  # Distortion Guitar -> Electric Guitar (jazz)
    31: 26,  # Guitar harmonics -> Electric Guitar (jazz)
    # ---------- Bass ----------
    32: 32,  # Acoustic Bass -> Acoustic Bass
    33: 33,  # Electric Bass (finger) -> Electric Bass
    34: 33,  # Electric Bass (pick) -> Electric Bass
    35: 33,  # Fretless Bass -> Electric Bass
    36: 36,  # Slap Bass 1 -> Slap Bass 1
    37: 36,  # Slap Bass 2 -> Slap Bass 1
    38: 38,  # Synth Bass 1 -> Synth Bass 1
    39: 38,  # Synth Bass 2 -> Synth Bass 1
    # ---------- Strings ----------
    40: 40,  # Violin -> Violin
    41: 41,  # Viola -> Viola
    42: 42,  # Cello -> Cello
    43: 43,  # Contrabass -> Contrabass
    44: 49,  # Tremolo Strings -> String Ensemble 1
    45: 49,  # Pizzicato Strings -> String Ensemble 1
    46: 46,  # Orchestral Harp -> Orchestral Harp
    47: 47,  # Timpani -> Timpani
    # ---------- Ensemble ----------
    48: 49,  # String Ensemble 1 -> String Ensemble 1
    49: 49,  # String Ensemble 2 -> String Ensemble 1
    50: 50,  # Synth Strings 1 -> Synth Strings
    51: 50,  # Synth Strings 2 -> Synth Strings
    52: 52,  # Choir Aahs -> Choir Aahs
    53: 52,  # Voice Oohs -> Choir Aahs
    54: 52,  # Synth Voice -> Choir Aahs
    55: 55,  # Orchestra Hit -> Orchestra Hit
    # ---------- Brass ----------
    56: 56,  # Trumpet -> Trumpet
    57: 57,  # Trombone -> Trombone
    58: 58,  # Tuba -> Tuba
    59: 56,  # Muted Trumpet -> Trumpet
    60: 60,  # French Horn -> French Horn
    61: 61,  # Brass Section -> Brass Section
    62: 62,  # Synth Brass 1 -> Synth Brass
    63: 62,  # Synth Brass 2 -> Synth Brass
    # ---------- Reed ----------
    64: 64,  # Soprano Sax -> Soprano Sax
    65: 65,  # Alto Sax -> Alto Sax
    66: 66,  # Tenor Sax -> Tenor Sax
    67: 67,  # Baritone Sax -> Baritone Sax
    68: 68,  # Oboe -> Oboe
    69: 69,  # English Horn -> English Horn
    70: 70,  # Bassoon -> Bassoon
    71: 71,  # Clarinet -> Clarinet
    # ---------- Pipe ----------
    72: 72,  # Piccolo -> Piccolo
    73: 73,  # Flute -> Flute
    74: 74,  # Recorder -> Recorder
    75: 75,  # Pan Flute -> Pan Flute
    76: None,  # Blown Bottle -> None
    77: None,  # Shakuhachi -> None
    78: None,  # Whistle -> None
    79: 79,  # Ocarina -> Ocarina
    # ---------- Synth Lead ----------
    80: 80,  # Lead 1 (square) -> Lead 1 (square)
    81: 80,  # Lead 2 (sawtooth) -> Lead 1 (square)
    82: 80,  # Lead 3 (calliope) -> Lead 1 (square)
    83: 80,  # Lead 4 (chiff) -> Lead 1 (square)
    84: 80,  # Lead 5 (charang) -> Lead 1 (square)
    85: 80,  # Lead 6 (voice) -> Lead 1 (square)
    86: 80,  # Lead 7 (fifths) -> Lead 1 (square)
    87: 80,  # Lead 8 (bass + lead) -> Lead 1 (square)
    # ---------- Synth Pad ----------
    88: 88,  # Pad 1 (new age) -> Pad 1 (new age)
    89: 88,  # Pad 2 (warm) -> Pad 1 (new age)
    90: 88,  # Pad 3 (polysynth) -> Pad 1 (new age)
    91: 88,  # Pad 4 (choir) -> Pad 1 (new age)
    92: 88,  # Pad 5 (bowed) -> Pad 1 (new age)
    93: 88,  # Pad 6 (metallic) -> Pad 1 (new age)
    94: 88,  # Pad 7 (halo) -> Pad 1 (new age)
    95: 88,  # Pad 8 (sweep) -> Pad 1 (new age)
    # ---------- Synth FX ----------
    96: None,  # FX 1 (rain) -> None
    97: None,  # FX 2 (soundtrack) -> None
    98: None,  # FX 3 (crystal) -> None
    99: None,  # FX 4 (atmosphere) -> None
    100: None,  # FX 5 (brightness) -> None
    101: None,  # FX 6 (goblins) -> None
    102: None,  # FX 7 (echoes) -> None
    103: None,  # FX 8 (sci-fi) -> None
    # ---------- Ethnic ----------
    104: 104,  # Sitar -> Sitar
    105: 105,  # Banjo -> Banjo
    106: 106,  # Shamisen -> Shamisen
    107: 107,  # Koto -> Koto
    108: 108,  # Koto -> Koto
    109: 109,  # Kalimba -> Kalimba
    110: 40,  # Bag pipe -> Synth Bass 2
    111: 111,  # Fiddle -> Fiddle
    # ---------- Percussive ----------
    112: None,  # Shanai -> None
    113: None,  # Tinkle Bell -> None
    114: None,  # Agogo -> None
    115: None,  # Steel Drums -> None
    116: None,  # Woodblock -> None
    117: 117,  # Taiko Drum -> Taiko Drum
    118: 118,  # Melodic Tom -> Melodic Tom
    119: 118,  # Synth Drum -> Melodic Tom
    # ---------- Sound FX ----------
    120: None,  # Reverse Cymbal -> None
    121: None,  # Guitar Fret Noise -> None
    122: None,  # Breath Noise -> None
    123: None,  # Seashore -> None
    124: None,  # Bird Tweet -> None
    125: None,  # Telephone Ring -> None
    126: None,  # Helicopter -> None
    127: None,  # Applause -> None
}
