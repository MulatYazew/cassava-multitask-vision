"""
AgroVision Africa — Loss Functions for Class Imbalance

Strategies for the severe class imbalance in the Cassava dataset (CMD ~61.5%, CBB ~5.1%):
  - WeightedCrossEntropyLoss : standard CE with per-class inverse-frequency weights
  - FocalLoss                : down-weights easy examples; focuses on hard minority cases
  - WeightedRandomSampler    : lives in data_handler.py; re-balances mini-batch composition

Avoid double-stacking corrections: if a WeightedRandomSampler is active, pass alpha=None
into the loss. Use the sampler OR class weights in the loss — not both.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


#  Weighted Cross-Entropy Loss

class WeightedCrossEntropyLoss(nn.Module):
    """
    Standard cross-entropy with per-class inverse-frequency weights and optional label smoothing.

    Args:
        weight          : Per-class weight tensor of shape ``(num_classes,)``.
                          Typically ``compute_class_weights(train_df).to(device)``.
                          ``None`` → all classes weighted equally (plain CE).
        label_smoothing : Label smoothing ε (default 0.0 = off).
                          0.1 is recommended for the noisy Cassava dataset labels.
        reduction       : ``'mean'`` (default) or ``'sum'``.
    """

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.register_buffer("weight", weight)
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        return F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction,
        )


#  Focal Loss

class FocalLoss(nn.Module):
    """
    Focal Loss.

    Down-weights easy / well-classified examples so the training signal
    is dominated by hard minority cases (especially CBB with only ~5 % of
    samples).

    Formula::

        FL(p_t) = -alpha_t · (1 - p_t)^gamma · log(p_t)

    where ``p_t`` is the probability assigned to the correct class.

    Args:
        alpha     : Per-class weight tensor of shape ``(num_classes,)``.
                    Pass ``compute_class_weights(train_df)`` here.
                    ``None`` → all classes weighted equally.
                    Skip (set ``None``) when a ``WeightedRandomSampler`` is
                    already active — see module docstring.
        gamma     : Focusing parameter.
                    ``0.0`` → reduces to standard (weighted) CE.
                    ``2.0`` → standard starting point (recommended).
                    ``3.0`` → raise if CBB F1 stays below 0.75 after 10 epochs.
        reduction : ``'mean'`` (default) or ``'sum'``.

    """

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        # Per-sample CE (possibly class-weighted + label-smoothed) — shape (B,)
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.alpha,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        # p_t = probability assigned to the correct class
        pt = torch.exp(-ce_loss)
        # Focal modulation: suppresses easy examples (high p_t → small weight)
        focal_loss = (1.0 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss  # reduction='none'


#  Factory 

def build_criterion(
    loss_type: str,
    class_weights: torch.Tensor | None,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> nn.Module:
    """
    Factory — return a configured loss module by name.

    Args:
        loss_type       : ``'focal'`` or ``'weighted_ce'``.
        class_weights   : Output of ``compute_class_weights(train_df).to(device)``.
                          Pass ``None`` when a ``WeightedRandomSampler`` is active
                          to avoid double-stacking frequency corrections.
        gamma           : Focal loss focusing parameter (ignored for ``weighted_ce``).
        label_smoothing : Smoothing ε; 0.1 recommended for noisy cassava labels.

    Returns:
        Configured loss module with weight buffer already registered.

    Example::

        # Sampler active — no alpha, with label smoothing
        criterion = build_criterion('focal', class_weights=None, gamma=2.0, label_smoothing=0.1)

        # No sampler — pass weights into loss
        criterion = build_criterion('focal', class_weights=w.to(device), gamma=2.0, label_smoothing=0.1)

        # Plain weighted CE with label smoothing
        criterion = build_criterion('weighted_ce', class_weights=w.to(device), label_smoothing=0.1)
    """
    if loss_type == "focal":
        return FocalLoss(alpha=class_weights, gamma=gamma, label_smoothing=label_smoothing)
    if loss_type == "weighted_ce":
        return WeightedCrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    raise ValueError(
        f"Unknown loss_type '{loss_type}'. "
        "Choose from: 'focal', 'weighted_ce'."
    )
