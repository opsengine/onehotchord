INSTRUMENTS = [
    0, 1, 2, 4, 5,      # Pianos
    24, 25, 26,         # Guitars
    40, 41, 42,         # Strings
]

SYNTH = [88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]

CHORD_DIR = "chords"
SOUND_FONT = "sf2/FluidR3_GM.sf2"

CHORDS_INTERVALS = {
    "dim": [0, 3, 6],
    "min": [0, 3, 7],
    "maj": [0, 4, 7],
    "7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    # "aug": [0, 4, 8],
    # "sus2": [0, 2, 7],
    # "sus4": [0, 5, 7],
    # "6": [0, 4, 7, 9],
}

# const chordNames = ["maj", "min", "7", "maj7", "min7", "dim", "aug", "sus2", "sus4", "6"];

CHORDS_BY_NAME = { chord: i for i, chord in enumerate(CHORDS_INTERVALS) }

CHORDS_BY_ID = { i: chord for i, chord in enumerate(CHORDS_INTERVALS) }

def invert_up(intervals):
    return [intervals[i+1] if i <len(intervals)-1 else intervals[0] + 12 for i in range(len(intervals))]

def invert_down(intervals):
    return [intervals[-1]-12 if i == 0 else intervals[i-1] for i in range(len(intervals))]
