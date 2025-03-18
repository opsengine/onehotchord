# One Hot Chord

An experimental deep learning project for real-time chord recognition from audio input. This project demonstrates how machine learning can be applied to music analysis. You can access the demo at https://onehotchord.com

> **Note:** This is an experimental project and may not work perfectly in all scenarios.

## Features

- Real-time chord recognition from audio input
- Browser-based interface with visual feedback
- Privacy-focused design - all processing happens on your device
- Supports common chord types: major, minor, diminished, 7th chords
- Responsive visualization of detected notes

## How It Works

One Hot Chord uses a deep neural network to analyze audio and identify chords:

1. Captures audio from your microphone
2. Extracts frequency information using a Constant-Q Transform
3. Processes these features through a neural network
4. Identifies the root note and chord type
5. Displays the results in real-time

## Project Structure

- `gen_samples.py` - Generates synthetic chord samples for training
- `preprocess.py` - Extracts features from audio samples
- `train.py` - Trains the neural network model
- `model.py` - Defines the neural network architecture
- `docs/` - Web demo

## Quick Start

### Prerequisites

- Python 3.8+
- FluidSynth and a SoundFont file (for sample generation)

### Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download a SoundFont file (e.g., FluidR3_GM.sf2) to the `sf2/` directory

### Training Pipeline

```bash
# Generate training samples
python gen_samples.py

# Preprocess audio samples
python preprocess.py [wav files]

# Train the model
python train.py chord_dataset.npz

# Try real-time recognition
python listen.py
```

### Web Interface

Serve the web interface:

```bash
cd www
python -m http.server 8000
```

Then open your browser to `http://localhost:8000`

## Limitations

This is still an experimental project built in a weekend. It has limitations:

- Works best with clean audio input
- May struggle with complex chord voicings
- Limited to the chord types it was trained on
- Performance varies depending on audio quality and background noise

## License

This project is licensed under the Apache License - see the LICENSE file for details.

## Acknowledgments

- [FluidSynth](https://www.fluidsynth.org/) for MIDI synthesis
- [Librosa](https://librosa.org/) for audio processing
- [PyTorch](https://pytorch.org/) for neural network implementation
- [ONNX Runtime](https://onnxruntime.ai/) for model deployment
