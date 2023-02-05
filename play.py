import numpy as np
import random
# might need to 'python3 -m pip install simpleaudio' or 'pip install simpleaudio' in terminal
import simpleaudio as sa
import matplotlib.pyplot as plt

NOTE_NAME_FREQUENCY = [
    ("C0", 16.35),#0
    ("C#0", 17.32),#1
    ("D0", 18.35),#2
    ("D#0", 19.45),#3
    ("E0", 20.60),#4
    ("F0", 21.83),#5
    ("F#0", 23.12),#6
    ("G0", 24.50),#7
    ("G#0", 25.96),#8
    ("A0", 27.50),#9
    ("A#0", 29.14),#10
    ("B0", 30.87),#11
    ("C1", 32.70),#12
    ("C#1", 34.65),#13
    ("D1", 36.71),#14
    ("D#1", 38.89),#15
    ("E1", 41.20),#16
    ("F1", 43.65),#17
    ("F#1", 46.25),#18
    ("G1", 49.00),#19
    ("G#1", 51.91),#20
    ("A1", 55.00),#21
    ("A#1", 58.27),#22
    ("B1", 61.74),#23
    ("C2", 65.41),#24
    ("C#2", 69.30),#25
    ("D2", 73.42),#26
    ("D#2", 77.78),#27
    ("E2", 82.41),#28
    ("F2", 87.31),#29
    ("F#2", 92.50),#30
    ("G2", 98.00),#31
    ("G#2", 103.83),#32
    ("A2", 110.00),#33
    ("A#2", 116.54),#34
    ("B2", 123.47),#35
    ("C3", 130.81),#36
    ("C#3", 138.59),#37
    ("D3", 146.83),#38
    ("D#3", 155.56),#39
    ("E3", 164.81),#40
    ("F3", 174.61),#41
    ("F#3", 185.00),#42
    ("G3", 196.00),#43
    ("G#3", 207.65),#44
    ("A3", 220.00),#45
    ("A#3", 233.08),#46
    ("B3", 246.94),#47
    ("C4", 261.63),#48
    ("C#4", 277.18),#49
    ("D4", 293.66),#50
    ("D#4", 311.13),#51
    ("E4", 329.63),#52
    ("F4", 349.23),#53
    ("F#4", 369.99),#54
    ("G4", 392.00),#55
    ("G#4", 415.30),#56
    ("A4", 440.00),#57
    ("A#4", 466.16),#58
    ("B4", 493.88),#59
    ("C5", 523.25),#60
    ("C#5", 554.37),#61
    ("D5", 587.33),#62
    ("D#5", 622.25),#63
    ("E5", 659.25),#64
    ("F5", 698.46),#65
    ("F#5", 739.99),#66
    ("G5", 783.99),#67
    ("G#5", 830.61),#68
    ("A5", 880.00),#69
    ("A#5", 932.33),#70
    ("B5", 987.77),#71
    ("C6", 1046.50),#72
    ("C#6", 1108.73),#73
    ("D6", 1174.66),#74
    ("D#6", 1244.51),#75
    ("E6", 1318.51),#76
    ("F6", 1396.91),#77
    ("F#6", 1479.98),#78
    ("G6", 1567.98),#79
    ("G#6", 1661.22),#80
    ("A6", 1760.00),#81
    ("A#6", 1864.66),#82
    ("B6", 1975.53),#83
    ("C7", 2093.00),#84
    ("C#7", 2217.46),#85
    ("D7", 2349.32),#86
    ("D#7", 2489.02),#87
    ("E7", 2637.02),#88
    ("F7", 2793.83),#89
    ("F#7", 2959.96),#90
    ("G7", 3135.96),#91
    ("G#7", 3322.44),#92
    ("A7", 3520.00),#93
    ("A#7", 3729.31),#94
    ("B7", 3951.07),#95
    ("C8", 4186.01),#96
    ("C#8", 4434.92),#97
    ("D8", 4698.63),#98
    ("D#8", 4978.03),#99
    ("E8", 5274.04),#100
    ("F8", 5587.65),#101
    ("F#8", 5919.91),#102
    ("G8", 6271.93),#103
    ("G#8", 6644.88),#104
    ("A8", 7040.00),#105
    ("A#8", 7458.62),#106
    ("B8", 7902.13)#107
]
NOTE_FREQUCIES = [name_note[1] for name_note in NOTE_NAME_FREQUENCY]
OCTAVE_SIZE = 12
SAMPLE_RATE = 44100  # 44100 samples per second

def get_note_index(note_name: str):
    for i, name_note in enumerate(NOTE_NAME_FREQUENCY):
        if name_note[0] == note_name:
            return i
    raise ValueError("Note name not found: {}".format(note_name))

def is_note_name(note_name: str):
    return any([name_note[0] == note_name for name_note in NOTE_NAME_FREQUENCY])

def avg_convolve_buffer(buffer, window_size: int):
    '''Smooths the audio buffer by averaging over a window of size window_size'''
    res = np.convolve(buffer, np.ones(window_size)/float(window_size), mode='same')
    return res

class Scale:
    def __init__(self, base_note, pattern):
        
        if not type(base_note) is int:
            if is_note_name(base_note):
                base_note = get_note_index(base_note)
            else:
                raise ValueError("Base note must be either an integer or a note name.")
        if any([type(p) is not int for p in pattern]):
            raise ValueError("Pattern must be a list of integers.")
        
        self.base_note = base_note
        self.pattern = pattern
        self.notes_per_octave = len(pattern)

    def __getitem__(self, index):
        idx_in_pattern = index%self.notes_per_octave
        base_relative_offset = self.pattern[idx_in_pattern]

        num_octaves_diff = index//self.notes_per_octave
        base_note_offset = num_octaves_diff * OCTAVE_SIZE
        
        global_idx = self.base_note + base_note_offset + base_relative_offset
        return NOTE_FREQUCIES[global_idx]

def generate_note(frequency: float, seconds: float = 0.5):
    # Generate array with seconds*sample_rate steps, ranging between 0 and seconds
    t = np.linspace(0.0, seconds, int(seconds * SAMPLE_RATE), False)
    note = np.sin(frequency * t * 2 * np.pi)
    return note

def convert_to_audio(notes):
    # Ensure that highest value is in 16-bit range
    audio = notes * (2**15 - 1) / np.max(np.abs(notes))
    # Convert to 16-bit data
    return audio.astype(np.int16)

def generate_sequential_notes(frequencies: list, durations: list = None):
    if type(durations) is float:
        durations = [durations] * len(frequencies)
    if durations is None:
        durations = [0.5] * len(frequencies)
    assert(len(frequencies) == len(durations))

    audios = []
    for frequency, seconds in zip(frequencies, durations):    
        audios.append(generate_note(frequency, seconds))

    return np.concatenate(audios)

def play_audio(audio):
    play_obj = sa.play_buffer(audio, 1, 2, SAMPLE_RATE)
    play_obj.wait_done()

def play_sequntial(frequencies: list, durations: list = None):
    notes = generate_sequential_notes(frequencies, durations)
    audio = convert_to_audio(notes)
    play_audio(audio)

def compose(scale: Scale, indices: list):
    return [scale[i] for i in indices]

HEPTATONIC_MAJOR_SCALE_PATTERN = [0, 2, 4, 5, 7, 9, 11]
PENTATONIC_BLUES_SCALE_PATTERN = [0, 3, 5, 6, 7, 10]
PENTATONIC_MAJOR_SCALE_PATTERN = [0, 2, 4, 7, 9]
HANDWRITTEN_SOLO_1 = [
    0, 1, 0, 0,
    1, 2, 3, 4,
    4, 3, 2, 1,
    0, 0, 0, 0,
    1, 2, 3, 4,
    3, 4, 3, 2,
    0, 2, 3, 4,
    0, 2, 3, 2,
    1, 2, 4, 5,
    6, 5, 6, 5,
    6, 8, 9, 8,
    6, 5, 6, 5,
    4, 3, 2, 1,
    1, 2, 3, 4,
    4, 3, 2, 1,
    0, 0, 0, 0,
]

def generate_sine_permutation_solo(time_step: float, total_time: float, baseline_periods: int, baseline_amplitude: int, permutation_max: int):
    num_samples = int(total_time/time_step)
    t = np.linspace(0.0, total_time, num_samples, False)
    baseline = np.sin(t*2*np.pi*baseline_periods/total_time)
    #convert to 0-BASELINE_APLITUDE range
    baseline = (baseline+1)*baseline_amplitude/2
    #convert to integer
    baseline = baseline.astype(np.int16)
    permutation = [random.randint(0, permutation_max) for _ in range(num_samples)]
    indices = [i + p for i, p in zip(baseline, permutation)]
    return indices

def expand_map(ls: list, f)->list:
    lss = [f(l) for l in ls]
    res = []
    for ls in lss:
        res.extend(ls)
    return res

TIME_STEP = 0.5
TOTAL_TIME = 60.0
SCALE_PATTERN = PENTATONIC_BLUES_SCALE_PATTERN
BASELINE_PERIODS = int(TOTAL_TIME/10)
BASELINE_AMPLITUDE = 7
PERMUTATION_MAX = 3
REVOICE = 3
indices_sets = [generate_sine_permutation_solo(TIME_STEP, TOTAL_TIME, BASELINE_PERIODS, BASELINE_AMPLITUDE, PERMUTATION_MAX)]
indices_sets = expand_map(indices_sets, lambda indices: [[i+j for i in indices] for j in range(2)])#convert each note to to triad
indices_sets = expand_map(indices_sets, lambda indices: [[i+j*len(SCALE_PATTERN) for i in indices] for j in range(3)])#duplicate each note over 2 octaves
# indices_sets = expand_map(indices_sets, lambda indices: [indices]*REVOICE)

tracks = [compose(Scale("C3", SCALE_PATTERN), indices) for indices in indices_sets]
notes_sets = [generate_sequential_notes(track, TIME_STEP) for track in tracks]
# notes_sets = [np.roll(notes, int(SAMPLE_RATE/TIME_STEP/(REVOICE*2)*i)) for i, notes in enumerate(notes_sets)]

notes = sum(notes_sets)
notes = avg_convolve_buffer(notes, 100)
# plt.plot(notes[:int(SAMPLE_RATE/TIME_STEP*5)])
# plt.show()
audio = convert_to_audio(notes)
play_audio(audio)
