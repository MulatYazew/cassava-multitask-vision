"""
Training loop for AgroVision project.
Handles model training, validation, and checkpointing.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
import time
from pathlib import Path
from typing import Tuple, Dict


class Trainer:
    """
    Trainer class for managing the training loop.
    """
    
    def __init__(self, model, device, learning_rate=0.001, weight_decay=1e-5):
        """
        Args:
            model: PyTorch model
            device: torch.device object
            learning_rate (float): Learning rate
            weight_decay (float): Weight decay for regularization
        """
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def train_epoch(self, train_loader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def validate(self, val_loader) -> Tuple[float, float]:
        """
        Validate the model.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
        
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, num_epochs=50, patience=5, 
              model_save_dir="models"):
        """
        Full training loop with early stopping.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs (int): Number of epochs
            patience (int): Early stopping patience
            model_save_dir (str): Directory to save checkpoints
        """
        Path(model_save_dir).mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # Train and validate
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Log progress
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # Save checkpoint
                checkpoint_path = f"{model_save_dir}/best_model.pth"
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"  [BEST] Model saved to {checkpoint_path}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            print()
        
        # Training summary
        elapsed_time = time.time() - start_time
        print(f"Training completed in {elapsed_time/60:.2f} minutes")
        
        return self.history
