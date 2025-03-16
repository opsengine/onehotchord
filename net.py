import torch
import torch.nn as nn
from common import CHORDS_BY_NAME

class OneHotChordNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        num_features = 24
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
