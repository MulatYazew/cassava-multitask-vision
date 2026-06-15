"""
AgroVision Africa — Loss Functions for Class Imbalance
=======================================================
Three complementary strategies address the severe class imbalance
in the Cassava dataset (CMD ≈ 61.5 %, CBB ≈ 5.1 %):

  1. ``WeightedCrossEntropyLoss`` — standard CE with per-class
     inverse-frequency weights.  Simple, no extra hyperparameters.

  2. ``FocalLoss`` — down-weights easy / well-classified examples
     so training focuses on hard minority cases, especially CBB.
     Combines class-frequency correction (alpha) with sample-difficulty
     correction (gamma).

  3. ``WeightedRandomSampler`` — lives in ``codes/data_handler.py``;
     re-balances mini-batch composition at the DataLoader level.

⚠️  IMPORTANT — avoid double-stacking corrections:
    If a ``WeightedRandomSampler`` is active in the DataLoader, pass
    ``alpha=None`` (and ``weight=None``) into the loss.  The sampler
    already equalises class frequency per batch; adding inverse-frequency
    weights on top amplifies minority classes *twice*, causing loss spikes
    or unstable training in early epochs when most predictions are "hard"
    and the (1-p_t)^gamma focal term is large.

    Choose ONE place to correct for class frequency:

      ┌────────────────────┬───────────────────────────────────────────────┐
      │ Sampler active     │ alpha=None, gamma free to tune                │
      │ No sampler         │ pass alpha/weight = compute_class_weights(…)  │
      └────────────────────┴───────────────────────────────────────────────┘

References:
    Lin et al. (2017). "Focal Loss for Dense Object Detection."
        https://arxiv.org/abs/1708.02002
    Cui et al. (2019). "Class-Balanced Loss Based on Effective Number of Samples."
        https://arxiv.org/abs/1901.05555
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


#  Weighted Cross-Entropy Loss

class WeightedCrossEntropyLoss(nn.Module):
    """
    Standard cross-entropy with per-class inverse-frequency weights.

    Advantages:
        Simple, no extra hyperparameters, numerically stable, well-understood.

    Disadvantages:
        Treats all examples within a class equally — does not distinguish
        hard from easy samples.

    Args:
        weight    : Per-class weight tensor of shape ``(num_classes,)``.
                    Typically ``compute_class_weights(train_df).to(device)``.
                    ``None`` → all classes weighted equally (plain CE).
        reduction : ``'mean'`` (default) or ``'sum'``.

    Usage::

        weights   = compute_class_weights(train_df).to(device)
        criterion = WeightedCrossEntropyLoss(weight=weights)
        loss      = criterion(logits, labels)
    """

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        # register_buffer moves the tensor with .to(device) automatically
        # and saves it in the state_dict for reproducibility.
        self.register_buffer("weight", weight)
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
            reduction=self.reduction,
        )


#  Focal Loss

class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017).

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

    Advantages:
        Simultaneously handles class imbalance (alpha) and sample difficulty
        (gamma).  Hard minority examples receive the largest gradient signal.

    Disadvantages:
        Two hyperparameters to tune.  Poorly chosen gamma can cause
        loss spikes on very small classes early in training.

    Usage::

        weights   = compute_class_weights(train_df).to(device)
        criterion = FocalLoss(alpha=weights, gamma=2.0)
        loss      = criterion(logits, labels)

        # With WeightedRandomSampler already active — omit alpha:
        criterion = FocalLoss(alpha=None, gamma=2.0)
    """

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.register_buffer("alpha", alpha)  # moves with .to(device)
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        # Per-sample CE (possibly class-weighted) — shape (B,)
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.alpha,
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
) -> nn.Module:
    """
    Factory — return a configured loss module by name.

    Args:
        loss_type     : ``'focal'`` or ``'weighted_ce'``.
        class_weights : Output of ``compute_class_weights(train_df).to(device)``.
                        Pass ``None`` when a ``WeightedRandomSampler`` is active
                        to avoid double-stacking frequency corrections.
        gamma         : Focal loss focusing parameter (ignored for ``weighted_ce``).

    Returns:
        Configured loss module with weight buffer already registered.

    Example::

        # Sampler active — no alpha in loss
        criterion = build_criterion('focal', class_weights=None, gamma=2.0)

        # No sampler — pass weights into loss
        criterion = build_criterion('focal', class_weights=w.to(device), gamma=2.0)

        # Plain weighted CE
        criterion = build_criterion('weighted_ce', class_weights=w.to(device))
    """
    if loss_type == "focal":
        return FocalLoss(alpha=class_weights, gamma=gamma)
    if loss_type == "weighted_ce":
        return WeightedCrossEntropyLoss(weight=class_weights)
    raise ValueError(
        f"Unknown loss_type '{loss_type}'. "
        "Choose from: 'focal', 'weighted_ce'."
    )
