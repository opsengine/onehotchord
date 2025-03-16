import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

from net import OneHotChordNet
from common import CHORDS_BY_NAME

# Helper functions
def load_and_prepare_data(data_path):
    """Load data and prepare train/validation datasets"""
    data = np.load(data_path)
    features = torch.tensor(data["features"], dtype=torch.float32)
    targets = torch.tensor(data["labels"], dtype=torch.float32)
    
    # Split into train/validation data
    train_idx, val_idx = train_test_split(np.arange(len(features)), test_size=0.2, random_state=42)
    
    train_features = features[train_idx]
    train_labels = targets[train_idx]
    
    val_features = features[val_idx]
    val_labels = targets[val_idx]
    
    # Create tensor datasets and dataloaders
    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    return train_loader, val_loader

def extract_targets(batch_targets):
    """Extract root, chord, and presence targets from batch"""
    root_targets = batch_targets[:, :12]
    num_chords = len(CHORDS_BY_NAME)
    chord_targets = batch_targets[:, 12:12+num_chords]
    presence_targets = batch_targets[:, 12+num_chords].unsqueeze(1)
    
    # Convert one-hot encoded targets to class indices for CrossEntropyLoss
    root_indices = torch.argmax(root_targets, dim=1)
    chord_indices = torch.argmax(chord_targets, dim=1)
    
    return root_targets, chord_targets, presence_targets, root_indices, chord_indices

def compute_classification_accuracy(outputs, targets):
    """Compute accuracy for classification (CrossEntropyLoss)"""
    _, predicted = torch.max(outputs, 1)
    targets_indices = torch.argmax(targets, dim=1)
    correct = (predicted == targets_indices).float()
    return correct.sum().item() / targets.size(0)

def compute_binary_accuracy(outputs, targets, threshold=0.5):
    """Compute accuracy for binary classification (BCEWithLogitsLoss)"""
    probs = torch.sigmoid(outputs)
    predicted = (probs > threshold).float()
    correct = (predicted == targets).float()
    return correct.sum().item() / targets.numel()

def calculate_loss(root_outputs, chord_outputs, presence_outputs, 
                  root_targets, chord_targets, presence_targets,
                  root_indices, chord_indices,
                  ce_criterion, bce_criterion):
    """Calculate combined loss based on presence of chord samples"""
    # Calculate presence loss (always)
    presence_loss = bce_criterion(presence_outputs, presence_targets)
    
    # Initialize total loss with presence loss
    loss = presence_loss
    
    # Add root and chord losses only if there are chord samples
    if presence_targets.sum() > 0:
        # Create a mask for samples with presence=1
        chord_mask = presence_targets.squeeze(1).bool()
        
        # Only if we have chord samples, calculate those losses
        if chord_mask.any():
            # Apply the mask to get only the chord samples
            masked_root_outputs = root_outputs[chord_mask]
            masked_root_indices = root_indices[chord_mask]
            
            masked_chord_outputs = chord_outputs[chord_mask]
            masked_chord_indices = chord_indices[chord_mask]
            
            # Calculate losses only on the chord samples
            root_loss = ce_criterion(masked_root_outputs, masked_root_indices)
            chord_loss = ce_criterion(masked_chord_outputs, masked_chord_indices)
            
            # Add to total loss
            loss += root_loss + chord_loss
    
    return loss

def calculate_accuracies(root_outputs, chord_outputs, presence_outputs,
                        root_targets, chord_targets, presence_targets):
    """Calculate accuracies for all outputs"""
    # For root and chord, only calculate accuracy on chord samples
    if presence_targets.sum() > 0:
        chord_mask = presence_targets.squeeze(1).bool()
        if chord_mask.any():
            root_acc = compute_classification_accuracy(root_outputs[chord_mask], root_targets[chord_mask])
            chord_acc = compute_classification_accuracy(chord_outputs[chord_mask], chord_targets[chord_mask])
        else:
            root_acc = 0
            chord_acc = 0
    else:
        root_acc = 0
        chord_acc = 0
    
    presence_acc = compute_binary_accuracy(presence_outputs, presence_targets)
    
    return root_acc, chord_acc, presence_acc

def train_epoch(model, train_loader, optimizer, ce_criterion, bce_criterion):
    """Train for one epoch"""
    model.train()
    train_loss = 0
    
    for features, targets in train_loader:
        optimizer.zero_grad()
        root_outputs, chord_outputs, presence_outputs = model(features)
        
        # Extract targets
        root_targets, chord_targets, presence_targets, root_indices, chord_indices = extract_targets(targets)
        
        # Calculate loss
        loss = calculate_loss(
            root_outputs, chord_outputs, presence_outputs,
            root_targets, chord_targets, presence_targets,
            root_indices, chord_indices,
            ce_criterion, bce_criterion
        )
        
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    return train_loss / len(train_loader)

def validate(model, val_loader, ce_criterion, bce_criterion):
    """Validate the model"""
    model.eval()
    val_loss = 0
    val_root_acc = 0
    val_chord_acc = 0
    val_presence_acc = 0
    val_batches = 0
    
    with torch.no_grad():
        for features, targets in val_loader:
            root_outputs, chord_outputs, presence_outputs = model(features)
            
            # Extract targets
            root_targets, chord_targets, presence_targets, root_indices, chord_indices = extract_targets(targets)
            
            # Calculate loss
            batch_loss = calculate_loss(
                root_outputs, chord_outputs, presence_outputs,
                root_targets, chord_targets, presence_targets,
                root_indices, chord_indices,
                ce_criterion, bce_criterion
            )
            
            val_loss += batch_loss.item()
            
            # Calculate accuracies
            root_acc, chord_acc, presence_acc = calculate_accuracies(
                root_outputs, chord_outputs, presence_outputs,
                root_targets, chord_targets, presence_targets
            )
            
            val_root_acc += root_acc
            val_chord_acc += chord_acc
            val_presence_acc += presence_acc
            val_batches += 1
    
    return (
        val_loss / len(val_loader),
        val_root_acc / val_batches,
        val_chord_acc / val_batches,
        val_presence_acc / val_batches
    )

def export_model(model, filename):
    """Export model to ONNX format"""
    model.eval()
    x = torch.randn(1, 24)
    
    torch.onnx.export(
        model, 
        x, 
        filename, 
        input_names=["input"], 
        output_names=["root_output", "chord_output", "presence_output"],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'root_output': {0: 'batch_size'},
            'chord_output': {0: 'batch_size'},
            'presence_output': {0: 'batch_size'}
        }
    )
    print(f"Model exported to {filename}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python train.py <data_file.npz>")
        sys.exit(1)
        
    # Load and prepare data
    train_loader, val_loader = load_and_prepare_data(sys.argv[1])
    
    # Initialize model and optimizer
    model = OneHotChordNet()
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # Define loss functions
    ce_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    bce_criterion = nn.BCEWithLogitsLoss()
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    # Variables to track best model
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_epoch = 0
    best_model_state = None  # Store the model state dict in memory
    
    # Training loop
    print("Training...")
    epochs = 200
    
    for epoch in range(epochs):
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, ce_criterion, bce_criterion)
        
        # Validate
        val_loss, val_root_acc, val_chord_acc, val_presence_acc = validate(
            model, val_loader, ce_criterion, bce_criterion
        )
        
        # Calculate combined accuracy (weighted average)
        combined_acc = (val_root_acc + val_chord_acc + val_presence_acc) / 3
        
        # Print metrics
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Root Acc: {val_root_acc:.4f}, Chord Acc: {val_chord_acc:.4f}, Presence Acc: {val_presence_acc:.4f}")
        
        # Save model state in memory if it's the best so far (by accuracy)
        if combined_acc > best_val_acc:
            best_val_acc = combined_acc
            best_val_loss = val_loss
            best_epoch = epoch + 1
            # Create a deep copy of the model's state dict
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"New best model! (Epoch {best_epoch})")
        
        # Update learning rate
        scheduler.step(val_loss)
    
    # Save the best model at the end of training
    if best_model_state is not None:
        torch.save(best_model_state, "best_one_hot_chord.pth")
        print(f"Best model saved from Epoch {best_epoch} with validation accuracy: {best_val_acc:.4f} and loss: {best_val_loss:.4f}")

    # rename the best model found
    os.rename("best_one_hot_chord.pth", "one_hot_chord.pth")
    
    # Load the best model for export
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model for export")
    
    # Export to ONNX
    export_model(model, "www/one_hot_chord.onnx")

if __name__ == "__main__":
    main()
