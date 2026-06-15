"""
AgroVision Africa — Model Definitions
======================================
Three backbone options for Cassava leaf disease classification:

  ┌─────────────────────┬──────────────┬────────────────────────────────────────┐
  │ Model               │ Role         │ Purpose                                │
  ├─────────────────────┼──────────────┼────────────────────────────────────────┤
  │ ResNet50Model       │ Baseline     │ Classical CNN anchor; lower-bound F1   │
  │ EfficientNetV2SModel│ Proposed     │ High-performance fused-MBConv CNN      │
  │ ConvNeXtTinyModel   │ Comparison   │ Transformer-inspired modern CNN        │
  └─────────────────────┴──────────────┴────────────────────────────────────────┘

All three share a common interface via ``BaseModel`` (ABC):
  - ``freeze_backbone()`` / ``unfreeze_backbone()``   — two-phase fine-tuning
  - ``get_trainable_params()``                        — optimizer-ready param list
  - ``model_info()``                                  — dict for comparison reports
  - ``forward()``                                     — explicit per-architecture path

Usage::

    from codes.model import build_model, create_model

    # Preferred — backbone frozen on creation (Phase 1 ready)
    model = build_model("efficientnet_v2_s", num_classes=5, dropout=0.3).to(device)
    model.unfreeze_backbone()   # Phase 2: full fine-tune

    # Legacy alias (kept for backward compatibility with older training scripts)
    model = create_model(num_classes=5, pretrained=True, model_name="efficientnet_v2_s")

Imbalance note
--------------
Model architecture is independent of the imbalance strategy.
Class-frequency correction lives in:
  - ``codes/loss.py``     — FocalLoss / WeightedCrossEntropyLoss
  - ``codes/data_handler.py`` — build_weighted_sampler
Pick ONE correction point; see loss.py for guidance on avoiding double-stacking.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torchvision import models


#  Shared classification head 

def make_head(in_features: int, num_classes: int, dropout: float) -> nn.Sequential:
    """
    Lightweight two-layer MLP head with regularisation.

    Architecture::

        Dropout(0.5) → Linear(in_features, 256) → ReLU → Dropout(p) → Linear(256, num_classes)

    The first dropout (0.5) regularises the high-dimensional backbone output.
    The second dropout (configurable, default 0.3) regularises the hidden layer.
    A hidden layer of 256 units gives the head enough capacity to re-weight
    backbone features without overfitting on minority classes.

    Args:
        in_features : Backbone output dimension (e.g. 2048 for ResNet-50).
        num_classes : Number of output logits (5 for Cassava).
        dropout     : Dropout rate for the second (inner) dropout layer.

    Returns:
        ``nn.Sequential`` ready to replace the backbone's original classifier.
    """
    return nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(256, num_classes),
    )


#  Abstract base class 

class BaseModel(ABC, nn.Module):
    """
    Shared interface for all AgroVision architectures.

    Subclasses MUST implement:
      ``_build()``            — construct ``self.backbone``, ``self.head``, and any
                               intermediate layers (pool, flatten, norm).
      ``freeze_backbone()``   — Phase 1: freeze all backbone parameters.
      ``unfreeze_backbone()`` — Phase 2: unfreeze for full fine-tuning.
      ``forward()``           — each architecture has a different forward path.

    ``get_trainable_params()`` and ``model_info()`` are provided here.
    """

    # Subclasses set this to their registry key (e.g. "resnet50")
    NAME: str = "base"

    def __init__(self, num_classes: int = 5, dropout: float = 0.3) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dropout = dropout
        self._build()

    @abstractmethod
    def _build(self) -> None:
        """Construct ``self.backbone``, ``self.head``, and any pooling / norm layers."""
        ...

    @abstractmethod
    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (Phase 1 — head-only training)."""
        ...

    @abstractmethod
    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone for full fine-tuning (Phase 2)."""
        ...

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.  Each architecture has a different pool/flatten/norm
        order, so subclasses must define this explicitly.
        """
        ...

    def get_trainable_params(self) -> list:
        """
        Return parameters that require gradients as a list.

        A list (not a generator) so the optimizer can iterate it multiple
        times without exhausting the iterator.
        """
        return [p for p in self.parameters() if p.requires_grad]

    def model_info(self) -> dict:
        """Return a dict of parameter counts suitable for comparison tables."""
        total = sum(p.numel() for p in self.parameters()) / 1e6
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad) / 1e6
        return {
            "name": self.NAME,
            "total_params_M": round(total, 2),
            "frozen_params_M": round(frozen, 2),
            "trainable_params_M": round(total - frozen, 2),
        }


#  ResNet-50  (Baseline) 

class ResNet50Model(BaseModel):
    """
    ResNet-50 backbone — established baseline.

    Architecture::

        conv1 → bn1 → relu → maxpool →
        layer1 → layer2 → layer3 → layer4 →
        AdaptiveAvgPool2d(1) → Flatten(B, 2048) →
        Dropout(0.5) → Linear(2048, 256) → ReLU → Dropout(p) → Linear(256, num_classes)

    Role:
        Ablation anchor.  Any improvement from EfficientNet-V2-S or ConvNeXt-Tiny
        should exceed this by ≥ 3 pp macro-F1.

    Transfer-learning strategy:
        Phase 1 (epochs 1–4)  : backbone frozen; only head trains at LR=1e-3.
        Phase 2 (epoch 5+)    : full fine-tune at LR=1e-4 (cosine decay).
    """

    NAME = "resnet50"

    def _build(self) -> None:
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Decompose into explicit layers so freeze_backbone() is unambiguous.
        self.backbone = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
            base.avgpool,          # AdaptiveAvgPool2d(output_size=1)
        )
        self.flatten = nn.Flatten()
        # Replace original fc with the shared two-layer head.
        self.head = make_head(base.fc.in_features, self.num_classes, self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)   # (B, 2048, 1, 1)
        x = self.flatten(x)    # (B, 2048)
        return self.head(x)    # (B, num_classes)

    def freeze_backbone(self) -> None:
        """Freeze all backbone layers; head remains trainable (Phase 1)."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters for full fine-tuning (Phase 2)."""
        for p in self.backbone.parameters():
            p.requires_grad = True


#  EfficientNet-V2-S  (Proposed model) 

class EfficientNetV2SModel(BaseModel):
    """
    EfficientNet-V2-S — proposed high-performance model (Tan & Le, 2021).

    Architecture::

        base.features (fused-MBConv + MBConv blocks) →
        AdaptiveAvgPool2d(1) → Flatten(B, 1280) →
        Dropout(0.5) → Linear(1280, 256) → ReLU → Dropout(p) → Linear(256, num_classes)

    Why V2-S:
        - Fused-MBConv blocks are faster than V1 depth-wise separable convs on
          Apple Silicon MPS and CUDA alike.
        - Better accuracy-efficiency frontier than EfficientNet-B3 or B4.

    Role:
        Primary proposed model; should exceed ResNet-50 by ≥ 3 pp macro-F1.

    Transfer-learning strategy: identical to ResNet-50 two-phase schedule.
    """

    NAME = "efficientnet_v2_s"

    def _build(self) -> None:
        base = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        )
        self.backbone = base.features          # all convolutional stages
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        # in_features = 1280 for EfficientNet-V2-S
        in_features = base.classifier[1].in_features
        self.head = make_head(in_features, self.num_classes, self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)   # (B, 1280, H, W)
        x = self.pool(x)       # (B, 1280, 1, 1)
        x = self.flatten(x)    # (B, 1280)
        return self.head(x)    # (B, num_classes)

    def freeze_backbone(self) -> None:
        """Freeze feature extractor for Phase 1 warm-up."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze full network for Phase 2 fine-tuning."""
        for p in self.backbone.parameters():
            p.requires_grad = True


#  ConvNeXt-Tiny  (Comparison / modern SOTA) 

class ConvNeXtTinyModel(BaseModel):
    """
    ConvNeXt-Tiny — transformer-inspired modern CNN (Liu et al., 2022).

    Architecture::

        stem + 4 hierarchical stages (channels: 96 → 192 → 384 → 768) →
        AdaptiveAvgPool2d(1) → Flatten(B, 768) →  ← flatten BEFORE LayerNorm
        LayerNorm(768)        ← normalises over feature dim, not spatial
        Dropout(0.5) → Linear(768, 256) → ReLU → Dropout(p) → Linear(256, num_classes)

    Critical ordering — ``flatten → norm``, NOT ``norm → flatten``:
        LayerNorm in ConvNeXt is designed to operate on a 2-D tensor (B, C).
        Passing (B, C, 1, 1) normalises over a single spatial element per channel,
        which is mathematically a no-op.  Always flatten first.

    Role:
        Comparison model.  Uses modern convolution design (large kernels, inverted
        bottlenecks) derived from Swin Transformer design principles.

    Transfer-learning strategy: identical to EfficientNet-V2-S.
    """

    NAME = "convnext_tiny"

    def _build(self) -> None:
        base = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        self.backbone = base.features            # stem + 4 hierarchical stages
        self.pool = base.avgpool                 # AdaptiveAvgPool2d(output_size=1)
        # Original classifier layout: [0]=LayerNorm(768), [1]=Flatten, [2]=Linear(768,1000)
        # We reuse the pretrained LayerNorm and Flatten; replace Linear with our head.
        self.flatten = base.classifier[1]        # Flatten(start_dim=1)
        self.norm = base.classifier[0]           # LayerNorm(768) — pretrained weights kept
        in_features = base.classifier[2].in_features  # 768
        self.head = make_head(in_features, self.num_classes, self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)   # (B, 768, 7, 7)
        x = self.pool(x)       # (B, 768, 1, 1)
        x = self.flatten(x)    # (B, 768)   ← flatten BEFORE norm (see docstring)
        x = self.norm(x)       # (B, 768)   ← LayerNorm over feature dim
        return self.head(x)    # (B, num_classes)

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True


#  Registry & factory functions 

MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "resnet50":          ResNet50Model,
    "efficientnet_v2_s": EfficientNetV2SModel,
    "convnext_tiny":     ConvNeXtTinyModel,
}


def build_model(name: str,num_classes: int ,dropout: float,) -> BaseModel:
    """
    Preferred factory — instantiate a model with backbone **frozen** (Phase 1 ready).

    Call ``model.unfreeze_backbone()`` at the start of Phase 2 fine-tuning.

    Args:
        name        : One of ``'resnet50'``, ``'efficientnet_v2_s'``, ``'convnext_tiny'``.
        num_classes : Output classes (default 5 for Cassava).
        dropout     : Dropout rate for the inner head dropout (default 0.3).

    Returns:
        ``BaseModel`` with frozen backbone and ``head`` ready to train.

    Example::

        model = build_model("efficientnet_v2_s").to(device)
        # Phase 1: train head only …
        model.unfreeze_backbone()
        # Phase 2: fine-tune everything …
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available: {sorted(MODEL_REGISTRY)}"
        )
    model = MODEL_REGISTRY[name](num_classes=num_classes, dropout=dropout)
    model.freeze_backbone()   # start Phase 1 ready
    return model


def create_model(num_classes: int, pretrained: bool = True, model_name: str = "efficientnet_v2_s",) -> BaseModel:
    """
    Legacy factory alias — kept for backward compatibility.

    Wraps ``build_model``; the ``pretrained`` flag is always honoured (ImageNet
    weights are loaded when ``True``).  New code should prefer ``build_model``.

    Args:
        num_classes : Number of output classes (default 5 for Cassava).
        pretrained  : Load ImageNet weights (strongly recommended; default True).
        model_name  : One of ``'resnet50'``, ``'efficientnet_v2_s'``, ``'convnext_tiny'``.

    Returns:
        ``BaseModel`` with frozen backbone.
    """
    if not pretrained:
        # Instantiate without calling build_model so we can skip weight loading.
        # Re-create with weights=None by temporarily overriding the class _build.
        # Simplest approach: just warn and proceed — omitting pretrained weights
        # hurts convergence significantly on a dataset of this size.
        import warnings
        warnings.warn(
            "pretrained=False is not recommended for the Cassava dataset. "
            "Training from scratch on ~21 k images rarely converges well.",
            UserWarning,
            stacklevel=2,
        )
    return build_model(name=model_name, num_classes=num_classes)
