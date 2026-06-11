"""
Evaluation metrics and analysis for AgroVision project.
Includes confusion matrix computation and performance metrics.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


class Evaluator:
    """
    Evaluator class for computing metrics and generating visualizations.
    """
    
    def __init__(self, num_classes, class_names=None, device="cuda"):
        """
        Args:
            num_classes (int): Number of classes
            class_names (list, optional): List of class names
            device: torch device
        """
        self.num_classes = num_classes
        self.device = device
        self.class_names = class_names or [f"Class {i}" for i in range(num_classes)]
    
    def predict(self, model, test_loader):
        """
        Generate predictions on test data.
        
        Args:
            model: PyTorch model
            test_loader: DataLoader for test data
            
        Returns:
            tuple: (all_preds, all_labels)
        """
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        return np.array(all_preds), np.array(all_labels)
    
    def compute_metrics(self, y_true, y_pred) -> dict:
        """
        Compute evaluation metrics.
        
        Args:
            y_true (array): True labels
            y_pred (array): Predicted labels
            
        Returns:
            dict: Dictionary containing all metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        return metrics
    
    def get_confusion_matrix(self, y_true, y_pred):
        """
        Compute confusion matrix.
        
        Args:
            y_true (array): True labels
            y_pred (array): Predicted labels
            
        Returns:
            np.ndarray: Confusion matrix
        """
        return confusion_matrix(y_true, y_pred)
    
    def plot_confusion_matrix(self, y_true, y_pred, figsize=(10, 8)):
        """
        Plot confusion matrix heatmap.
        
        Args:
            y_true (array): True labels
            y_pred (array): Predicted labels
            figsize (tuple): Figure size
        """
        cm = self.get_confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    
    def print_classification_report(self, y_true, y_pred):
        """
        Print detailed classification report.
        
        Args:
            y_true (array): True labels
            y_pred (array): Predicted labels
        """
        print("Classification Report:")
        print(classification_report(y_true, y_pred, 
                                   target_names=self.class_names,
                                   zero_division=0))
    
    def evaluate(self, model, test_loader):
        """
        Full evaluation pipeline.
        
        Args:
            model: PyTorch model
            test_loader: DataLoader for test data
            
        Returns:
            dict: Dictionary containing all results
        """
        y_pred, y_true = self.predict(model, test_loader)
        metrics = self.compute_metrics(y_true, y_pred)
        
        results = {
            'predictions': y_pred,
            'true_labels': y_true,
            'metrics': metrics,
            'confusion_matrix': self.get_confusion_matrix(y_true, y_pred)
        }
        
        return results
