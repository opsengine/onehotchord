import torch
from model import OneHotChordNet
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split


model = OneHotChordNet()
model.load_state_dict(torch.load("one_hot_chord.pth"))
model.eval()

data = np.load("chord_dataset.npz")
features = torch.tensor(data["features"], dtype=torch.float32)
targets = torch.tensor(data["labels"], dtype=torch.float32)

# split into train/validation data
train_idx, val_idx = train_test_split(np.arange(len(features)), test_size=0.2, random_state=42)

train_features = features[train_idx]
train_labels = targets[train_idx]

val_features = features[val_idx]
val_labels = targets[val_idx]

# create tensor datasets
train_dataset = TensorDataset(train_features, train_labels)
val_dataset = TensorDataset(val_features, val_labels)

val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
with torch.no_grad():
    for features, targets in val_loader:
        outputs = model(features)
        print(features)
        print(outputs)
        print(targets)
        break


# val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
# with torch.no_grad():
#     for i, (input_data, root_label, type_label) in enumerate(val_loader):
#         if i >= 3: break
#         root_pred, type_pred = model(input_data)
#         print(f"Sample {i+1}:")
#         print(f"Root Pred: {root_pred.numpy()[0].round(2)}")
#         print(f"Root True: {root_label.numpy()[0]}")
#         print(f"Type Pred: {type_pred.numpy()[0].round(2)}")
#         print(f"Type True: {type_label.numpy()[0]}")