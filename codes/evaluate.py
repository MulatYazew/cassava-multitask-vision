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

    def _predict_probs(
        self, model: nn.Module, loader: torch.utils.data.DataLoader
    ) -> tuple[torch.Tensor, np.ndarray]:
        """Return softmax probability matrix (N, C) and true labels for a loader."""
        model.eval()
        probs_list, labels_list = [], []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                probs  = torch.softmax(model(images), dim=1)
                probs_list.append(probs.cpu())
                labels_list.extend(labels.numpy())
        return torch.cat(probs_list, dim=0), np.array(labels_list)

    def predict(self, model: nn.Module, loader: torch.utils.data.DataLoader) -> tuple[np.ndarray, np.ndarray]:
        """Run model on loader; return (predictions, true_labels)."""
        probs, labels = self._predict_probs(model, loader)
        return probs.argmax(1).numpy(), labels

    def predict_tta(
        self,
        model:       nn.Module,
        loader:      torch.utils.data.DataLoader,
        n_augments:  int = 5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Test-Time Augmentation (TTA) inference.

        Runs the loader ``n_augments`` times, averaging softmax outputs.
        The loader must be constructed with ``augment=True`` so that each pass
        applies a different random augmentation to every image.

        Improves prediction confidence by averaging over multiple augmented views.

        Args:
            model      : Trained model in eval mode.
            loader     : DataLoader built from a ``CassavaDataset(augment=True)``.
            n_augments : Number of augmented passes to average (default 5).

        Returns:
            (predictions, true_labels) as numpy arrays.

        Example::

            tta_loader = DataLoader(CassavaDataset(test_df, img_dir, augment=True), batch_size=32)
            preds, labels = evaluator.predict_tta(model, tta_loader, n_augments=5)
        """
        all_probs: list[torch.Tensor] = []
        true_labels: np.ndarray | None = None

        for i in range(n_augments):
            probs, labels = self._predict_probs(model, loader)
            all_probs.append(probs)
            if true_labels is None:
                true_labels = labels

        avg_probs = torch.stack(all_probs).mean(0)
        return avg_probs.argmax(1).numpy(), true_labels  # type: ignore[return-value]

    def predict_ensemble(
        self,
        models: list[nn.Module],
        loader: torch.utils.data.DataLoader,
        use_tta: bool = False,
        n_augments: int = 5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Ensemble inference: average softmax outputs across multiple models.

        Combining models with different inductive biases (e.g. CNN + attention)
        typically yields stronger and more robust predictions.

        Args:
            models     : List of trained models (e.g. [effnet_model, swin_model]).
            loader     : DataLoader for the evaluation set.
            use_tta    : If True, apply TTA for each model (loader must have augment=True).
            n_augments : TTA steps per model (ignored when use_tta=False).

        Returns:
            (predictions, true_labels) as numpy arrays.

        Example::

            preds, labels = evaluator.predict_ensemble(
                [effnet, swin], test_loader, use_tta=True, n_augments=5
            )
        """
        all_probs: list[torch.Tensor] = []
        true_labels: np.ndarray | None = None

        for model in models:
            if use_tta:
                preds, labels = self.predict_tta(model, loader, n_augments=n_augments)
                # Reconstruct probs from TTA for averaging — rerun to get probs directly
                model_probs: list[torch.Tensor] = []
                for _ in range(n_augments):
                    probs, lbl = self._predict_probs(model, loader)
                    model_probs.append(probs)
                    if true_labels is None:
                        true_labels = lbl
                all_probs.append(torch.stack(model_probs).mean(0))
            else:
                probs, labels = self._predict_probs(model, loader)
                all_probs.append(probs)
                if true_labels is None:
                    true_labels = labels

        avg_probs = torch.stack(all_probs).mean(0)
        return avg_probs.argmax(1).numpy(), true_labels  # type: ignore[return-value]

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