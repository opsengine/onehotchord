import torch
import torch.nn as nn
import numpy as np
from common import CHORDS_BY_NAME

NUM_FEATURES = 27

class OneHotChordNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        num_features = NUM_FEATURES
        num_roots = 12
        num_chords = len(CHORDS_BY_NAME)

        self.shared_layers = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Output layer for root note prediction
        self.root_output = nn.Linear(64, num_roots)
        # Output layer for chord prediction
        self.chord_output = nn.Linear(64, num_chords)
        # Output layer for chord presence prediction
        self.presence_output = nn.Linear(64, 1)

    def forward(self, x):
        x = self.shared_layers(x)
        root_logits = self.root_output(x)
        chord_logits = self.chord_output(x)
        presence_logits = self.presence_output(x)

        return root_logits, chord_logits, presence_logits


def infer(model, features):
    if isinstance(features, list):
        features = np.array(features, dtype=np.float32)

    # Ensure features is the right shape (add batch dimension if needed)
    if len(features.shape) == 1:
        features = features.reshape(1, -1)

    # Convert to tensor
    x = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        root_logits, chord_logits, presence_logits = model(x)
        # Apply softmax to get probabilities
        root_probs = torch.softmax(root_logits, dim=1)
        chord_probs = torch.softmax(chord_logits, dim=1)
        presence_prob = torch.sigmoid(presence_logits).item()

    # Find the predicted classes
    root_pred = torch.argmax(root_probs, dim=1).item()
    chord_pred = torch.argmax(chord_probs, dim=1).item()

    return root_pred, chord_pred, presence_prob
