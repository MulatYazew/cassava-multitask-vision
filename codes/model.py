"""
AgroVision Africa — Model Definitions

Three backbone options for cassava leaf disease classification:
  - ResNet50Model        : classical CNN baseline
  - EfficientNetV2SModel : proposed high-performance model
  - SwinTinyModel        : Vision Transformer with LoRA-adapted attention

All share the BaseModel interface (freeze/unfreeze backbone, model_info).
Recommended: EfficientNetV2S + SwinTiny ensemble for best results.

    from codes.model import build_model
    model = build_model("efficientnet_v2_s", num_classes=5).to(device)
    model.unfreeze_backbone()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import math

import torch
import torch.nn as nn
from torchvision import models


# -- shared head -------------------------------------------------------------

def make_head(in_features: int, num_classes: int = 5, dropout: float = 0.3) -> nn.Sequential:
    return nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(256, num_classes),
    )


# -- abstract base -----------------------------------------------------------

class BaseModel(ABC, nn.Module):
    """Shared interface for all AgroVision backbones."""

    NAME: str = "base"

    def __init__(self, num_classes: int = 5, dropout: float = 0.3) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dropout = dropout
        self.build()

    @abstractmethod
    def build(self) -> None: ...

    @abstractmethod
    def freeze_backbone(self) -> None: ...

    @abstractmethod
    def unfreeze_backbone(self) -> None: ...

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    def get_trainable_params(self) -> list:
        return [p for p in self.parameters() if p.requires_grad]

    def model_info(self) -> dict:
        total  = sum(p.numel() for p in self.parameters()) / 1e6
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad) / 1e6
        return {
            "name":               self.NAME,
            "total_params_M":     round(total, 2),
            "frozen_params_M":    round(frozen, 2),
            "trainable_params_M": round(total - frozen, 2),
        }


# -- ResNet-50 ---------------------------------------------------------------

class ResNet50Model(BaseModel):
    """ResNet-50 baseline. Two-phase fine-tuning: head-only then full network."""

    NAME = "resnet50"

    def build(self) -> None:
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
            base.avgpool,
        )
        self.flatten = nn.Flatten()
        self.head = make_head(base.fc.in_features, self.num_classes, self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.flatten(self.backbone(x)))

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters(): p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters(): p.requires_grad = True


# -- EfficientNet-V2-S -------------------------------------------------------

class EfficientNetV2SModel(BaseModel):
    """
    EfficientNet-V2-S — proposed model.

    Fused-MBConv blocks give better throughput than V1 depth-wise convolutions
    and a better accuracy-efficiency frontier than EfficientNet-B3/B4.
    """

    NAME = "efficientnet_v2_s"

    def build(self) -> None:
        base = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        )
        self.backbone = base.features
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.flatten  = nn.Flatten()
        self.head     = make_head(base.classifier[1].in_features, self.num_classes, self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.flatten(self.pool(self.backbone(x))))

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters(): p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters(): p.requires_grad = True


# -- LoRA utilities ----------------------------------------------------------

class LoRALinear(nn.Module):
    """
    Wraps nn.Linear with a trainable low-rank delta:  out = (W + B·A·scale)·x

    W is frozen; B is zero-initialized so the output is unchanged at init.
    """

    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        in_f, out_f  = linear.in_features, linear.out_features
        self.linear  = linear
        self.scaling = alpha / rank

        for p in self.linear.parameters():
            p.requires_grad = False

        self.lora_A = nn.Parameter(torch.empty(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight + (self.lora_B @ self.lora_A) * self.scaling

    @property
    def bias(self) -> torch.Tensor | None:
        return self.linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        return F.linear(x, self.weight, self.bias)

    def merge_weights(self) -> nn.Linear:
        merged = nn.Linear(
            self.linear.in_features, self.linear.out_features,
            bias=self.linear.bias is not None,
            device=self.linear.weight.device,
            dtype=self.linear.weight.dtype,
        )
        merged.weight.data = self.weight.detach().clone()
        if self.linear.bias is not None:
            merged.bias.data = self.linear.bias.data.clone()
        return merged


def inject_lora(module: nn.Module, rank: int, alpha: float, targets: tuple[str, ...]) -> int:
    count = 0
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and name in targets:
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha))
            count += 1
        else:
            count += inject_lora(child, rank, alpha, targets)
    return count


def merge_lora(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, LoRALinear):
            setattr(module, name, child.merge_weights())
        else:
            merge_lora(child)


# -- Swin Transformer Tiny + LoRA --------------------------------------------

class SwinTinyModel(BaseModel):
    """
    Swin-Tiny fine-tuned with LoRA.

    LoRA injects rank-r matrices alongside frozen attention Linears. With rank=8
    this adds ~0.3% extra params while matching full fine-tuning on small datasets.

    Phase 1: only LoRA matrices + head (~0.5 M params).
    Phase 2: full backbone + LoRA + head.
    """

    NAME = "swin_tiny"

    def __init__(
        self,
        num_classes:  int             = 5,
        dropout:      float           = 0.3,
        lora_rank:    int             = 8,
        lora_alpha:   float           = 16.0,
        lora_targets: tuple[str, ...] = ("qkv", "proj"),
    ) -> None:
        self.lora_rank    = lora_rank
        self.lora_alpha   = lora_alpha
        self.lora_targets = lora_targets
        super().__init__(num_classes=num_classes, dropout=dropout)

    def build(self) -> None:
        base = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        self.backbone = base.features
        self.norm     = base.norm
        self.permute  = base.permute
        self.pool     = base.avgpool
        self.flatten  = nn.Flatten()
        self.head     = make_head(base.head.in_features, self.num_classes, self.dropout)
        inject_lora(self.backbone, self.lora_rank, self.lora_alpha, self.lora_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.norm(x)
        x = self.permute(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.head(x)

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters(): p.requires_grad = False
        for p in self.norm.parameters():     p.requires_grad = False
        for m in self.backbone.modules():
            if isinstance(m, LoRALinear):
                m.lora_A.requires_grad = True
                m.lora_B.requires_grad = True

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters(): p.requires_grad = True
        for p in self.norm.parameters():     p.requires_grad = True

    def merge_lora(self) -> None:
        merge_lora(self.backbone)

    def model_info(self) -> dict:
        info = super().model_info()
        lora_params = sum(
            p.numel()
            for m in self.backbone.modules()
            if isinstance(m, LoRALinear)
            for p in [m.lora_A, m.lora_B]
        ) / 1e6
        info.update({"lora_rank": self.lora_rank, "lora_alpha": self.lora_alpha,
                     "lora_params_M": round(lora_params, 4)})
        return info


# -- registry & factories ----------------------------------------------------

MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "resnet50":          ResNet50Model,
    "efficientnet_v2_s": EfficientNetV2SModel,
    "swin_tiny":         SwinTinyModel,
}


def build_model(name: str, num_classes: int = 5, dropout: float = 0.3, **kwargs) -> BaseModel:
    """
    Instantiate a model with backbone frozen (Phase 1 ready).
    Call model.unfreeze_backbone() to begin Phase 2.

    Args:
        name        : 'resnet50' | 'efficientnet_v2_s' | 'swin_tiny'
        num_classes : output classes (default 5 for Cassava)
        dropout     : inner head dropout rate
        **kwargs    : forwarded to model constructor (e.g. lora_rank for swin_tiny)
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    model = MODEL_REGISTRY[name](num_classes=num_classes, dropout=dropout, **kwargs)
    model.freeze_backbone()
    return model


def create_model(
    num_classes: int  = 5,
    pretrained:  bool = True,
    model_name:  str  = "efficientnet_v2_s",
) -> BaseModel:
    """Legacy alias for build_model. Prefer build_model in new code."""
    if not pretrained:
        import warnings
        warnings.warn(
            "pretrained=False is not recommended. Training from scratch on ~21k images rarely converges.",
            UserWarning, stacklevel=2,
        )
    return build_model(name=model_name, num_classes=num_classes)
