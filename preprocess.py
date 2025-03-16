import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import re
import argparse
import glob
import sys
import random
from scipy.signal import lfilter
import soundfile as sf

from common import *

def extract_features(audio_buffer, sr):
    octaves = 5
    C = librosa.cqt(
        audio_buffer, 
        sr=sr, 
        n_bins=octaves * 12 + 1,
        bins_per_octave=12,
        fmin=librosa.note_to_hz('C2'),
        hop_length=512,
        filter_scale=4.0,
    )

    # Convert to magnitude
    mag = np.abs(C)
    # average over time
    mag = np.mean(mag, axis=1)
    # group by octave
    reshaped = mag[:-1].reshape(octaves, 12)
    
    # octave ids (start from C2)
    octave_indices = np.arange(octaves)[:, np.newaxis] + 2

    # Calculate the denominator for frequency centroid
    denominator = np.sum(reshaped, axis=0)
    if np.any(denominator == 0):
        raise ValueError("Frequency centroid denominator is zero")

    # frequency centroid for each note, normalized for 7 octaves
    freq_centroid = np.sum(reshaped * octave_indices, axis=0) / denominator / 7

    # total amplitude for each note across all octaves
    chroma = np.sum(reshaped, axis=0)
    max_chroma = np.max(chroma)
    if max_chroma == 0:
        raise ValueError("Chroma is all zeros")

    # Sum across octaves and normalize
    normalized_chroma = chroma / max_chroma

    # return features
    return np.concatenate([normalized_chroma, freq_centroid])

def add_noise(audio_buffer, noise_level=0.005, noise='white'):
    if noise is None:
        return audio_buffer
    
    # Make a copy of the audio buffer to avoid modifying the original
    audio_buffer = np.copy(audio_buffer)
    
    # Get the RMS amplitude of the signal
    signal_rms = np.sqrt(np.mean(audio_buffer**2))
    
    # Scale noise by the signal RMS and the noise level parameter
    noise_amplitude = signal_rms * noise_level
    
    # Different noise types
    if noise == 'white':
        # White noise (equal energy per frequency)
        noise = np.random.normal(0, noise_amplitude, len(audio_buffer))
    elif noise == 'pink':
        # Pink noise (1/f spectrum - more energy in lower frequencies)
        white_noise = np.random.normal(0, noise_amplitude, len(audio_buffer))
        # Simple approximation of pink noise using a low-pass filter
        b, a = [0.049922035, -0.095993537, 0.050612699, -0.004408786], [1, -2.494956002, 2.017265875, -0.522189400]
        noise = lfilter(b, a, white_noise)
    elif noise == 'brownian':
        # Brownian noise (1/f^2 spectrum - even more energy in lower frequencies)
        noise = np.cumsum(np.random.normal(0, noise_amplitude / 10, len(audio_buffer)))
        # Normalize to have the same RMS as white noise would
        noise = noise * noise_amplitude / np.sqrt(np.mean(noise**2) + 1e-10)  # Add small epsilon to avoid division by zero
    elif noise == 'gaussian':
        # Gaussian noise with random frequency emphasis
        noise = np.random.normal(0, noise_amplitude, len(audio_buffer))
        # Apply random frequency emphasis
        n_fft = 2048
        S = librosa.stft(noise, n_fft=n_fft)
        emphasis = np.random.uniform(0.5, 2.0, S.shape[0])
        S_emphasized = S * emphasis[:, np.newaxis]
        noise = librosa.istft(S_emphasized, length=len(audio_buffer))
    else:
        raise ValueError(f"Unknown noise type: {noise}")
    
    # Add noise to the signal
    result = audio_buffer + noise
    
    # Check for and fix any non-finite values
    if not np.all(np.isfinite(result)):
        # Replace non-finite values with zeros
        result[~np.isfinite(result)] = 0
        print("Warning: Non-finite values detected and replaced with zeros")
    
    return result

def process_file(file_path):
    """
    Process a single file and return its features and label
    """
    # Extract label from filename (format: root_chordtype_instrument.wav)
    filename = os.path.basename(file_path)
    match = re.match(r'(\d+)_([a-z0-9]+)_.*\.wav', filename)
    if match:
        chord_present = True
        root = int(match.group(1))
        chord_type = match.group(2)
    elif re.match(r'noise_([a-z]+)_(\d+\.\d+)_(\d+)\.wav', filename):
        chord_present = False
        # bogus values. won't be used for training
        root = 0
        chord_type = 0
    elif re.match(r'single_(\d+)_(\d+)\.wav', filename):
        # Single note files are not chords
        chord_present = False
        # bogus values. won't be used for training
        root = 0
        chord_type = 0
    else:
        raise ValueError(f"Could not parse filename format: {filename}, expected root_chordtype_xxx.wav")
    
    if chord_present:
        if root not in range(12):
            raise ValueError(f"Root {root} is out of range, expected 0-11")
        if chord_type not in CHORDS_BY_NAME:
            valid_chords = ' '.join(list(CHORDS_BY_NAME.keys()))
            raise ValueError(f"Chord type {chord_type} not found. Valid types: {valid_chords}")

    one_hot_root = np.zeros(12)
    one_hot_chord = np.zeros(len(CHORDS_BY_NAME))
    if chord_present:
        one_hot_root[root] = 1
        one_hot_chord[CHORDS_BY_NAME[chord_type]] = 1

    # Load audio file
    audio_buffer, sr = librosa.load(file_path, sr=None)
    print(f"File: {os.path.basename(file_path)}, Length: {len(audio_buffer)} samples, Duration: {len(audio_buffer)/sr:.3f} seconds, Sample rate: {sr}")

    # Process the audio
    for noise in [None, 'white', 'pink', 'brownian', 'gaussian']:
        audio_buffer = add_noise(audio_buffer, noise_level=random.uniform(0, 0.3), noise=noise)
        try:
            features = extract_features(audio_buffer, sr)
            labels = np.concatenate([one_hot_root, one_hot_chord, np.array([chord_present])])
            yield (features, labels)
        except Exception as e:
            print(f"Warning: Error processing {filename} with noise type {noise}: {e}")
            continue

def process_file_safe(file_path):
    try:
        return (True, list(process_file(file_path)))
    except Exception as e:
        raise
        return (False, file_path, str(e))

def process_files_parallel(file_paths, output_file, n_workers=None):
    """
    Process specified audio files in parallel and save to a single file
    """
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    print(f"Processing {len(file_paths)} WAV files with {n_workers} workers")
    
    features = []
    labels = []
    
    # Process files in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(process_file_safe, file_paths))
    
    # Separate successful and failed results
    successful_results = []
    failed_files = []

    # Process the results
    for result in results:
        if result[0]:  # Success
            # result[1] is a list of (feature, label) pairs from one file
            file_results = result[1]
            
            # Iterate through each pair from this file
            for feature, label in file_results:
                if feature is not None and label is not None:
                    features.append(feature)
                    labels.append(label)
        else:  # Failure
            failed_files.append((result[1], result[2]))

    # Report failed files
    if failed_files:
        print(f"\nFailed to process {len(failed_files)} files:")
        for file_path, error in failed_files:
            print(f"  - {file_path}: {error}")
    
    # Convert to numpy arrays
    features = np.array(features)
    labels = np.array(labels)
    
    # Save to file
    np.savez(output_file, features=features, labels=labels)
    print(f"Saved {len(features)} samples to {output_file}")
    print(f"Dataset shape: {features.shape}")
    
    return features, labels

def parse_arguments():
    parser = argparse.ArgumentParser(description='Preprocess audio files for chord recognition.')
    parser.add_argument('input', nargs='+', help='Input files or glob patterns (e.g., "chords/*.wav" or "chords/11_*.wav")')
    parser.add_argument('-o', '--output', default='chord_dataset.npz', help='Output NPZ file (default: chord_dataset.npz)')
    parser.add_argument('-w', '--workers', type=int, default=None, help='Number of worker processes (default: CPU count)')
    parser.add_argument('-p', '--plot', action='store_true', help='Plot examples after processing')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Expand glob patterns and collect all file paths
    file_paths = []
    for pattern in args.input:
        matched_files = glob.glob(pattern)
        if not matched_files:
            print(f"Warning: No files match pattern '{pattern}'")
        file_paths.extend(matched_files)
    
    if not file_paths:
        print("Error: No input files found. Please check your file paths or patterns.")
        sys.exit(1)
    
    # Process files in parallel and save to a single file
    features, labels = process_files_parallel(file_paths, args.output, args.workers)
    
    # Print some stats about the dataset
    print(f"Number of samples: {len(features)}")
    print(f"Feature dimension: {features.shape}")
    
    # Optional: Plot a few examples
    if args.plot and len(features) > 0:
        plt.figure(figsize=(12, 8))
        for i in range(min(5, len(features))):
            plt.subplot(5, 1, i+1)
            plt.plot(features[i])
            plt.title(f"Root: {labels[i][0]}, Chord: {labels[i][1]}")
        plt.tight_layout()
        plt.show()
