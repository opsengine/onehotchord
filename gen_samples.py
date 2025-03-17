import mido
import subprocess
import os
import random
import numpy as np
from pydub import AudioSegment
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import wave
import struct

from common import *

SAMPLE_RATE = 22050

def get_instrument():
    return random.choice(INSTRUMENTS)


def get_random_velocity():
    return random.randint(80, 120)

def generate_chord_midi(root, intervals, filename, duration, instrument):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    tempo = mido.bpm2tempo(60)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
    track.append(mido.Message('program_change', program=instrument, time=0))

    root_midi = 48 + root

    for note in [root_midi + i for i in intervals]:
        track.append(mido.Message('note_on', note=note, velocity=get_random_velocity(), time=0))

    first = True
    for note in [root_midi + i for i in intervals]:
        if first:
            track.append(mido.Message('note_off', note=note, velocity=0, time=int(duration * 480)))
            first = False
        else:
            track.append(mido.Message('note_off', note=note, velocity=0, time=0))
    
    track.append(mido.MetaMessage('end_of_track', time=0))
    mid.save(filename)


def truncate_wav(wav_file, duration):
    audio = AudioSegment.from_wav(wav_file)
    audio = audio[:duration * 1000]
    audio.export(wav_file, format="wav")


def generate_chord_wav(root, chord_type, intervals, duration, instrument, invert):
    filename = f"{CHORD_DIR}/{root}_{chord_type}_{instrument}_{invert}"
    midi_file = f"{filename}.mid"
    output_file = f"{filename}.wav"
    
    for i in range(invert):
        intervals = invert_up(intervals)

    generate_chord_midi(root, intervals, midi_file, duration, instrument)
    
    gain = random.uniform(0.2, 0.5)
    subprocess.run([
        'fluidsynth',
        '--no-midi-in',
        '--no-shell',
        f'--sample-rate={SAMPLE_RATE}',
        '--reverb=0',
        '--chorus=0',
        f'--gain={gain}',
        '--fast-render', output_file,
        '--audio-file-type=wav',
        '--quiet',
        SOUND_FONT,
        midi_file
    ])
    os.remove(midi_file)
    truncate_wav(output_file, 1)
    print(f"Generated {output_file}")
    return output_file

# New functions for generating noise samples

def generate_white_noise(output_file, duration=1.0, amplitude=0.5):
    """Generate white noise WAV file"""
    sample_rate = SAMPLE_RATE
    num_samples = int(duration * sample_rate)
    
    # Generate white noise
    noise = np.random.uniform(-1, 1, num_samples) * amplitude
    
    # Convert to 16-bit PCM
    noise = (noise * 32767).astype(np.int16)
    
    # Write to WAV file
    with wave.open(output_file, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        for sample in noise:
            wav_file.writeframes(struct.pack('h', sample))
    
    print(f"Generated white noise: {output_file}")
    return output_file

def generate_pink_noise(output_file, duration=1.0, amplitude=0.5):
    """Generate pink noise WAV file"""
    sample_rate = SAMPLE_RATE
    num_samples = int(duration * sample_rate)
    
    # Generate white noise
    white_noise = np.random.uniform(-1, 1, num_samples)
    
    # Convert to pink noise using Voss algorithm (simplified)
    pink_noise = np.zeros(num_samples)
    
    # Number of octaves
    num_octaves = 5
    
    # Generate pink noise
    for i in range(num_octaves):
        # Calculate octave samples, ensuring it's at least 1
        octave_samples = max(1, num_samples // (2 ** i))
        pink_octave = np.random.uniform(-1, 1, octave_samples)
        
        # Use proper resizing to ensure exact length match
        if octave_samples < num_samples:
            # Repeat and truncate to exact length
            repeats = int(np.ceil(num_samples / octave_samples))
            pink_octave = np.tile(pink_octave, repeats)[:num_samples]
        else:
            # Truncate if somehow larger
            pink_octave = pink_octave[:num_samples]
        
        # Now pink_octave is guaranteed to be exactly num_samples in length
        pink_noise += pink_octave
    
    # Normalize and apply amplitude
    pink_noise = pink_noise / num_octaves
    pink_noise = pink_noise * amplitude
    
    # Convert to 16-bit PCM
    pink_noise = (pink_noise * 32767).astype(np.int16)
    
    # Write to WAV file
    with wave.open(output_file, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        for sample in pink_noise:
            wav_file.writeframes(struct.pack('h', sample))
    
    print(f"Generated pink noise: {output_file}")
    return output_file

def generate_brown_noise(output_file, duration=1.0, amplitude=0.5):
    """Generate brown noise WAV file"""
    sample_rate = SAMPLE_RATE
    num_samples = int(duration * sample_rate)
    
    # Generate brown noise using random walk
    brown_noise = np.zeros(num_samples)
    brown_noise[0] = np.random.uniform(-0.01, 0.01)
    
    for i in range(1, num_samples):
        step = np.random.uniform(-0.01, 0.01)
        brown_noise[i] = brown_noise[i-1] + step
    
    # Normalize and apply amplitude
    brown_noise = brown_noise / np.max(np.abs(brown_noise))
    brown_noise = brown_noise * amplitude
    
    # Convert to 16-bit PCM
    brown_noise = (brown_noise * 32767).astype(np.int16)
    
    # Write to WAV file
    with wave.open(output_file, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        for sample in brown_noise:
            wav_file.writeframes(struct.pack('h', sample))
    
    print(f"Generated brown noise: {output_file}")
    return output_file

def generate_sine_sweep(output_file, duration=1.0, amplitude=0.5):
    """Generate a sine sweep from 20Hz to 10kHz"""
    sample_rate = SAMPLE_RATE
    num_samples = int(duration * sample_rate)
    
    # Generate time array
    t = np.linspace(0, duration, num_samples)
    
    # Logarithmic frequency sweep from 20Hz to 10kHz
    start_freq = 20
    end_freq = 10000
    
    # Calculate instantaneous frequency
    freq = start_freq * np.exp(t * np.log(end_freq/start_freq)/duration)
    
    # Generate sine sweep
    phase = 2 * np.pi * np.cumsum(freq) / sample_rate
    sweep = np.sin(phase) * amplitude
    
    # Convert to 16-bit PCM
    sweep = (sweep * 32767).astype(np.int16)
    
    # Write to WAV file
    with wave.open(output_file, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        for sample in sweep:
            wav_file.writeframes(struct.pack('h', sample))
    
    print(f"Generated sine sweep: {output_file}")
    return output_file

def generate_noise_sample(noise_type, amplitude, index):
    """Generate a noise sample of the specified type and amplitude"""
    noise_dir = f"{CHORD_DIR}/noise"
    os.makedirs(noise_dir, exist_ok=True)
    
    output_file = f"{noise_dir}/noise_{noise_type}_{amplitude:.2f}_{index}.wav"
    
    if noise_type == "white":
        return generate_white_noise(output_file, amplitude=amplitude)
    elif noise_type == "pink":
        return generate_pink_noise(output_file, amplitude=amplitude)
    elif noise_type == "brown":
        return generate_brown_noise(output_file, amplitude=amplitude)
    elif noise_type == "sweep":
        return generate_sine_sweep(output_file, amplitude=amplitude)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

def process_noise(args):
    noise_type, amplitude, index = args
    return generate_noise_sample(noise_type, amplitude, index)

def process_chord(args):
    root, chord_type, intervals, duration, instrument, invert = args
    return generate_chord_wav(root, chord_type, intervals, duration, instrument, invert)

def generate_single_note_midi(note, filename, duration, instrument):
    """Generate a MIDI file with a single note"""
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    tempo = mido.bpm2tempo(60)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
    track.append(mido.Message('program_change', program=instrument, time=0))

    # Add the single note
    track.append(mido.Message('note_on', note=note, velocity=get_random_velocity(), time=0))
    track.append(mido.Message('note_off', note=note, velocity=0, time=int(duration * 480)))
    
    track.append(mido.MetaMessage('end_of_track', time=0))
    mid.save(filename)

def generate_single_note_wav(note_value, duration, instrument):
    """Generate a WAV file with a single note"""
    note_dir = f"{CHORD_DIR}/single_notes"
    os.makedirs(note_dir, exist_ok=True)
    
    filename = f"{note_dir}/single_{note_value}_{instrument}"
    midi_file = f"{filename}.mid"
    output_file = f"{filename}.wav"
    
    # MIDI note values: C4 is 60, range from 36 (C2) to 96 (C7)
    generate_single_note_midi(note_value, midi_file, duration, instrument)
    
    subprocess.run([
        'fluidsynth',
        '--no-midi-in',
        '--no-shell',
        '--sample-rate=' + str(SAMPLE_RATE),
        '--reverb=0',
        '--chorus=0',
        '--gain=0.5',
        '--fast-render', output_file,
        '--audio-file-type=wav',
        '--quiet',
        SOUND_FONT,
        midi_file
    ])
    os.remove(midi_file)
    print(f"Generated single note: {output_file}")
    return output_file

def process_single_note(args):
    """Process function for parallel generation of single notes"""
    note_value, duration, instrument = args
    return generate_single_note_wav(note_value, duration, instrument)

if __name__ == "__main__":
   
    os.makedirs(CHORD_DIR, exist_ok=True)

    duration = 0.9
    # Create a list of all tasks to process
    chord_tasks = []
    for root in range(12):
        for chord_type, intervals in CHORDS_INTERVALS.items():
            for instrument in INSTRUMENTS:
                for invert in range(4):
                    chord_tasks.append((root, chord_type, intervals, duration, instrument, invert))
    
    # Create a list of noise tasks
    noise_tasks = []
    noise_types = ["white", "pink", "brown", "sweep"]
    amplitudes = [0.01, 0.2, 0.5, 0.9]  # Different noise levels
    
    # Generate 10 samples of each noise type at each amplitude
    for noise_type in noise_types:
        for amplitude in amplitudes:
            for index in range(10):
                noise_tasks.append((noise_type, amplitude, index))
    
    # Create a list of single note tasks (negative samples)
    single_note_tasks = []
    # MIDI note values from C2 (36) to B6 (95)
    for note_value in range(36, 96):
        for instrument in INSTRUMENTS:
            single_note_tasks.append((note_value, duration, instrument))
    
    # Number of workers (adjust based on your CPU)
    num_workers = mp.cpu_count()
    
    # Process chord tasks in parallel
    print("Generating chord samples...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        chord_results = list(executor.map(process_chord, chord_tasks))
    
    # Process noise tasks in parallel
    print("Generating noise samples...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        noise_results = list(executor.map(process_noise, noise_tasks))
    
    # Process single note tasks in parallel
    print("Generating single note samples...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        single_note_results = list(executor.map(process_single_note, single_note_tasks))
