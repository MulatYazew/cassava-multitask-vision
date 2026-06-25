"""
AgroVision Grad-CAM

Gradient-weighted Class Activation Mapping for the three backbones in model.py
(CassavaCNN, EfficientNet-V2-S, Swin-Tiny).

    from codes.gradcam import GradCAM, overlay_heatmap, show_gradcam

    cam = GradCAM(model, model_name=MODEL_NAME)
    heatmap, pred_class = cam(input_tensor)   # input_tensor: (1, C, H, W)
    overlay = overlay_heatmap(original_rgb_image, heatmap)

Supported model_name values: cassava_cnn, efficientnet_v2_s, swin_tiny.
"""

from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .data_handler import IMAGENET_MEAN, IMAGENET_STD, get_transforms, CLASS_NAMES


#  Target-layer lookup per backbone 

def get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """
    Return the last convolutional block to hook, per architecture.

      cassava_cnn       — model.stage4[-1] : last ResBlock+SE in stage4
      efficientnet_v2_s — backbone[-1]     : last MBConv block in base.features
      swin_tiny         — model.permute    : already (B, 768, H, W); backbone[-1]
                                             would give (B, H, W, C) which breaks GAP
    """
    if model_name == "swin_tiny":
        # model.permute converts (B, H, W, 768) → (B, 768, H, W).
        # Hooking here keeps activations/gradients in CAM-compatible (B, C, H, W) order.
        return model.permute

    if model_name == "cassava_cnn":
        # stage4 is an nn.Sequential of ResBlock+SE units; [-1] is the last one.
        # Its output is a (B, 256, H, W) spatial feature map — ideal for CAM.
        return model.stage4[-1]

    backbone = model.backbone

    if model_name == "efficientnet_v2_s":
        # backbone IS base.features (Sequential of MBConv stages).
        return backbone[-1]

    raise ValueError(
        f"Unknown model_name '{model_name}'. "
        f"Expected one of: cassava_cnn, efficientnet_v2_s, swin_tiny."
    )


#  Grad-CAM 

class GradCAM:
    """
    Grad-CAM wrapper for the AgroVision models.

    Args:
        model        : trained nn.Module (one of the three AgroVision backbones).
        model_name   : 'cassava_cnn' | 'efficientnet_v2_s' | 'swin_tiny'
        target_layer : explicit layer to hook; auto-detected from model_name if None.
    """

    def __init__(self, model: nn.Module, model_name: str, target_layer: Optional[nn.Module] = None) -> None:
        self.model = model
        self.model.eval()

        self.target_layer = target_layer or get_target_layer(model, model_name)

        self.stored_activations: Optional[torch.Tensor] = None
        self.stored_gradients:   Optional[torch.Tensor] = None

        self.fwd_hook = self.target_layer.register_forward_hook(self.save_activation)
        self.bwd_hook = self.target_layer.register_full_backward_hook(self.save_gradient)

    #  Hooks

    def save_activation(self, module, inp, out) -> None:
        self.stored_activations = out.detach()

    def save_gradient(self, module, grad_in, grad_out) -> None:
        self.stored_gradients = grad_out[0].detach()

    #  Core

    def __call__(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> tuple[np.ndarray, int]:
        """
        Args:
            input_tensor : (1, C, H, W), already normalized (e.g. via get_transforms).
            class_idx    : class to explain. If None, uses the model's own prediction.

        Returns:
            heatmap   : (H, W) float32 array in [0, 1], resized to input resolution.
            class_idx : the class index used.
        """
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        input_tensor.requires_grad_(True)

        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        self.model.zero_grad(set_to_none=True)
        score = logits[0, class_idx]
        score.backward(retain_graph=False)

        activations = self.stored_activations[0]      # (C, h, w)
        gradients   = self.stored_gradients[0]         # (C, h, w)

        weights = gradients.mean(dim=(1, 2))           # (C,) — spatial mean per channel

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for c, w in enumerate(weights):
            cam += w * activations[c]

        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        h, w = input_tensor.shape[-2:]
        heatmap = cam.cpu().numpy()
        heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        return heatmap, class_idx

    def remove_hooks(self) -> None:
        self.fwd_hook.remove()
        self.bwd_hook.remove()


#  Visualization helpers 

def overlay_heatmap(rgb_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on an RGB image.

    Args:
        rgb_image : (H, W, 3) uint8 array (0-255), already resized to heatmap size.
        heatmap   : (H, W) float array in [0, 1].
        alpha     : heatmap opacity.

    Returns:
        (H, W, 3) uint8 blended image.
    """
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    if rgb_image.shape[:2] != heatmap.shape:
        rgb_image = cv2.resize(rgb_image, (heatmap.shape[1], heatmap.shape[0]))

    blended = (alpha * colored + (1 - alpha) * rgb_image).astype(np.uint8)
    return blended


def preprocess_for_gradcam(rgb_image: np.ndarray, image_size: int) -> tuple[torch.Tensor, np.ndarray]:
    """
    Apply the standard (no-augmentation) eval transform from data_handler.get_transforms,
    plus produce a resized RGB uint8 copy for overlaying.

    Returns:
        input_tensor : (1, C, H, W) normalized tensor.
        resized_rgb  : (image_size, image_size, 3) uint8 array for display.
    """
    transform = get_transforms(image_size, augment=False)
    tensor = transform(image=rgb_image)["image"].unsqueeze(0)
    resized_rgb = cv2.resize(rgb_image, (image_size, image_size))
    return tensor, resized_rgb


def show_gradcam(
    model: nn.Module,
    model_name: str,
    image: np.ndarray,
    image_size: int,
    class_names: Optional[dict] = None,
    device: Optional[torch.device] = None,
    true_label: Optional[int] = None,
    class_idx: Optional[int] = None,
    figsize: tuple[int, int] = (12, 4),
) -> None:
    """
    End-to-end Grad-CAM visualization for a single RGB image.

    Args:
        model       : trained model (already on `device`, in eval mode).
        model_name  : 'cassava_cnn' | 'efficientnet_v2_s' | 'swin_tiny'.
        image       : (H, W, 3) RGB uint8 array.
        image_size  : INPUT_SIZE from config.py.
        class_names : CLASS_NAMES dict from data_handler (default used if None).
        device      : torch.device; if None, taken from model parameters.
        true_label  : optional ground-truth class index, shown in the title.
        class_idx   : explain this class instead of the model's prediction.
    """
    class_names = class_names or CLASS_NAMES
    device = device or next(model.parameters()).device

    input_tensor, resized_rgb = preprocess_for_gradcam(image, image_size)

    cam = GradCAM(model, model_name=model_name)
    heatmap, used_class = cam(input_tensor.to(device), class_idx=class_idx)
    cam.remove_hooks()

    overlay = overlay_heatmap(resized_rgb, heatmap)

    # Predicted class & confidence (for the title)
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(input_tensor.to(device)), dim=1)[0].cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    axes[0].imshow(resized_rgb); axes[0].set_title("Input"); axes[0].axis("off")
    axes[1].imshow(heatmap, cmap="jet"); axes[1].set_title("Grad-CAM"); axes[1].axis("off")
    axes[2].imshow(overlay); axes[2].set_title("Overlay"); axes[2].axis("off")

    title = f"Explained class: {class_names.get(used_class, used_class)} ({probs[used_class]*100:.1f}%)"
    if true_label is not None:
        title += f"  |  True: {class_names.get(true_label, true_label)}"
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.show()


def show_gradcam_grid(
    model: nn.Module,
    model_name: str,
    images: list[np.ndarray],
    image_size: int,
    class_names: Optional[dict] = None,
    device: Optional[torch.device] = None,
    true_labels: Optional[list[int]] = None,
    cols: int = 4,
    figsize_per_cell: float = 3.0,
) -> None:
    """
    Grid of Grad-CAM overlays for multiple images (e.g. failure cases
    from per_class_f1_report / confusion-matrix error analysis).
    """
    class_names = class_names or CLASS_NAMES
    device = device or next(model.parameters()).device
    n = len(images)
    rows = max(1, (n + cols - 1) // cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * figsize_per_cell, rows * figsize_per_cell))
    axes = np.array(axes).reshape(-1)

    cam = GradCAM(model, model_name=model_name)
    model.eval()

    for i, img in enumerate(images):
        input_tensor, resized_rgb = preprocess_for_gradcam(img, image_size)
        heatmap, used_class = cam(input_tensor.to(device))
        overlay = overlay_heatmap(resized_rgb, heatmap)

        ax = axes[i]
        ax.imshow(overlay)
        title = f"Pred: {class_names.get(used_class, used_class)}"
        if true_labels is not None:
            title += f"\nTrue: {class_names.get(true_labels[i], true_labels[i])}"
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    cam.remove_hooks()
    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()
