import torch
import numpy as np
import pyaudio
import time
import queue
import librosa

from net import OneHotChordNet
from common import CHORDS_BY_NAME
from preprocess import extract_features

# Constants
SAMPLE_RATE = 22050
WINDOW_SIZE = int(SAMPLE_RATE * 0.5)  # 500ms window
STEP_SIZE = int(SAMPLE_RATE * 0.2)    # 100ms step
BUFFER_SIZE = WINDOW_SIZE * 2         # Buffer size (2x window size)
FORMAT = pyaudio.paFloat32
CHANNELS = 1

# Note names and chord types
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
CHORD_NAMES = list(CHORDS_BY_NAME.keys())

# Load the trained model
def load_model(model_path="one_hot_chord.pth"):
    model = OneHotChordNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Predict chord from features
def predict_chord(model, features):
    # Convert to PyTorch tensor - fix the warning by ensuring features is a numpy array
    if isinstance(features, list):
        features = np.array(features, dtype=np.float32)
    
    # Ensure features is the right shape (add batch dimension if needed)
    if len(features.shape) == 1:
        features = features.reshape(1, -1)
        
    # Convert to tensor
    x = torch.tensor(features, dtype=torch.float32)
    
    # Forward pass
    with torch.no_grad():
        root_logits, chord_logits, presence_logits = model(x)
        
        # Apply softmax to get probabilities
        root_probs = torch.softmax(root_logits, dim=1)
        chord_probs = torch.softmax(chord_logits, dim=1)
        presence_prob = torch.sigmoid(presence_logits).item()

    # Find the predicted classes
    root_pred = torch.argmax(root_probs, dim=1).item()
    chord_pred = torch.argmax(chord_probs, dim=1).item()
    
    # Get confidence scores
    root_conf = root_probs[0, root_pred].item()
    chord_conf = chord_probs[0, chord_pred].item()

    # if presence_prob > 0.5:
    #     # print features and probabilities
    #     print(f"\n\nFeatures: {features}")
    #     print(f"Root probabilities: {root_probs}")
    #     print(f"Chord probabilities: {chord_probs}")
    #     print(f"Presence probability: {presence_prob}")

    return root_pred, chord_pred, root_conf, chord_conf, presence_prob

# Audio callback function
def audio_callback(in_data, frame_count, time_info, status):
    audio_queue.put(np.frombuffer(in_data, dtype=np.float32))
    return (in_data, pyaudio.paContinue)

# Main function
def main():
    global audio_queue
    audio_queue = queue.Queue()
    
    # Load model
    print("Loading model...")
    model = load_model()
    print("Model loaded successfully")
    
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    
    # Open audio stream
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=STEP_SIZE,
        stream_callback=audio_callback
    )
    
    print("Listening for chords... (Press Ctrl+C to stop)")
    
    # Start the stream
    stream.start_stream()
    
    # Buffer to store audio data
    audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
    
    # Minimum confidence threshold
    confidence_threshold = 0.3
    presence_threshold = 0.5
    
    try:
        while stream.is_active():
            # Get audio data from queue
            if not audio_queue.empty():
                audio_data = audio_queue.get()
                
                # Update buffer (shift old data left, add new data to the right)
                audio_buffer = np.roll(audio_buffer, -len(audio_data))
                audio_buffer[-len(audio_data):] = audio_data
                
                # Extract features from the last WINDOW_SIZE samples
                features = extract_features(audio_buffer[-WINDOW_SIZE:], SAMPLE_RATE)
                
                # Predict chord
                root_pred, chord_pred, root_conf, chord_conf, presence_prob = predict_chord(model, features)
                
                chord_name = f"{NOTE_NAMES[root_pred]}{CHORD_NAMES[chord_pred]}"
                if presence_prob > presence_threshold: # and root_conf > confidence_threshold and chord_conf > confidence_threshold:
                    print(f"Detected: {chord_name} (Root: {root_conf*100:.1f}%, Chord: {chord_conf*100:.1f}%, Presence: {presence_prob*100:.1f}%)   ")
                # else:
                #     print(f"No chord detected: {chord_name} (Root: {root_conf*100:.1f}%, Chord: {chord_conf*100:.1f}%, Presence: {presence_prob*100:.1f}%)   ")
                # else:
                #     print("\rNo chord detected with high confidence                                ", end="")
            
            # Sleep to reduce CPU usage
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\nStopping...")
    
    finally:
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Audio stream closed")

if __name__ == "__main__":
    main()