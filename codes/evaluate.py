"""
AgroVision Evaluator
Inference, metrics, confusion matrix, and per-class F1 report.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)


class Evaluator:
    """
    Computes metrics and visualizations for a trained model.

    Args:
        num_classes  : Number of output classes.
        class_names  : List of class-name strings (length == num_classes).
        device       : torch.device used for inference.
    """

    def __init__(
        self,
        num_classes:  int,
        class_names:  list[str] | None = None,
        device:       torch.device     = torch.device("cpu"),
    ) -> None:
        self.num_classes = num_classes
        self.device      = device
        self.class_names = class_names or [f"Class {i}" for i in range(num_classes)]

    #  Inference 

    def predict(self, model: nn.Module, loader: torch.utils.data.DataLoader) -> tuple[np.ndarray, np.ndarray]:
        """Run model on loader; return (predictions, true_labels)."""
        model.eval()
        preds_list, labels_list = [], []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                preds  = model(images).argmax(1)
                preds_list.extend(preds.cpu().numpy())
                labels_list.extend(labels.numpy())
        return np.array(preds_list), np.array(labels_list)

    #  Metrics 

    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Return accuracy, weighted precision/recall/F1."""
        return {
            "accuracy":  accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall":    recall_score(   y_true, y_pred, average="weighted", zero_division=0),
            "f1":        f1_score(       y_true, y_pred, average="weighted", zero_division=0),
        }

    def print_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Print sklearn classification report."""
        print(classification_report(y_true, y_pred, target_names=self.class_names, zero_division=0))

    #  Confusion matrix 

    def plot_confusion_matrix(
        self,
        y_true:  np.ndarray,
        y_pred:  np.ndarray,
        figsize: tuple[int, int] = (10, 8),
        normalize: bool = False,
    ) -> None:
        """
        Plot a confusion matrix heatmap.

        Args:
            normalize : If True, normalize by true-label counts (shows rates).
        """
        cm = confusion_matrix(y_true, y_pred)
        fmt = ".2f"
        if normalize:
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        else:
            fmt = "d"

        plt.figure(figsize=figsize)
        sns.heatmap(
            cm, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={"label": "Rate" if normalize else "Count"},
        )
        plt.title("Confusion Matrix" + (" (normalized)" if normalize else ""))
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    #  Full pipeline 

    def evaluate(self, model: nn.Module, loader: torch.utils.data.DataLoader) -> dict:
        """
        End-to-end evaluation: predict → metrics → confusion matrix.

        Returns dict with keys: predictions, true_labels, metrics, confusion_matrix.
        """
        y_pred, y_true = self.predict(model, loader)
        return {
            "predictions":    y_pred,
            "true_labels":    y_true,
            "metrics":        self.compute_metrics(y_true, y_pred),
            "confusion_matrix": confusion_matrix(y_true, y_pred),
        }