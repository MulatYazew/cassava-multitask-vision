"""
AgroVision Data Handler
Dataset, DataLoader, augmentation pipelines, and class-weight helpers.
Apple Silicon note: num_workers=0 avoids macOS multiprocessing hangs.
"""

import json
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, WeightedRandomSampler


#  Class metadata 

CLASS_NAMES: dict[int, str] = {
    0: "Cassava Bacterial Blight (CBB)",
    1: "Cassava Brown Streak Disease (CBSD)",
    2: "Cassava Green Mottle (CGM)",
    3: "Cassava Mosaic Disease (CMD)",
    4: "Healthy",
}

CLASS_DESCRIPTIONS: dict[int, str] = {
    0: "Bacterial infection causing angular leaf spots and wilting.",
    1: "Viral disease with yellow/brown streaks along leaf veins.",
    2: "Viral disease with mottled green patches on leaves.",
    3: "Viral disease causing mosaic patterns and leaf distortion.",
    4: "No visible disease symptoms detected.",
}

# Classes with <3 000 samples → heavier augmentation.
MINORITY_CLASSES: set[int] = {0, 1, 2}   # CBB (~1 100), CBSD (~2 200), CGM (~2 400)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


#  Label-map loader 

def load_label_map(dataset_dir: str | Path) -> dict[int, str]:
    """Load the label map JSON if present, else fall back to CLASS_NAMES."""
    path = Path(dataset_dir) / "label_num_to_disease_map.json"
    if not path.exists():
        return CLASS_NAMES
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


#  Augmentation pipelines 

def get_transforms(image_size: int = 224, augment: bool = True) -> A.Compose:
    """
    Standard pipeline for majority classes and validation/inference.

    Augmentations are chosen for African field conditions:
    - RandomResizedCrop  : varying camera distance from leaf
    - HorizontalFlip     : no preferred left/right orientation
    - Rotate             : field photos are rarely perfectly upright
    - BrightnessContrast : lighting varies significantly across Africa
    - ColorJitter        : white-balance differences in cheap smartphones
    - Affine             : small translations add spatial diversity
    """
    if augment:
        return A.Compose([
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=25, border_mode=0, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),   # field-lighting variation
            A.Affine(translate_percent=0.05, scale=(0.90, 1.10), rotate=(-15, 15), p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_robust_transforms(image_size: int = 224 ) -> A.Compose:
    """
    Heavy augmentation pipeline for minority classes (CBB, CBSD, CGM).
    Extra ops vs the standard pipeline:
    - VerticalFlip       : no canonical top/bottom for cassava leaves
    - GaussianBlur       : motion blur / low-quality smartphone optics
    - ImageCompression   : JPEG artefacts from compressed cheap devices
    - CoarseDropout      : partial occlusion by dust, fingers, overlapping leaves
    - RandomShadow       : shadows cast by canopy vegetation
    """
    return A.Compose([
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.5, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.4),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=35, border_mode=0, p=0.6),
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.6),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.7),
        A.Affine(translate_percent=0.08, scale=(0.85, 1.15), rotate=(-20, 20), p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.4),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),  # cheap-camera sensor noise
        A.ImageCompression(quality_range=(60, 95), p=0.3),
        A.CoarseDropout(
            num_holes_range=(4, 8),
            hole_height_range=(16, 32),
            hole_width_range=(16, 32),
            p=0.4,
        ),
        A.RandomShadow(p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


#  Dataset 

class CassavaDataset(Dataset):
    """
    PyTorch Dataset for cassava leaf images.

    Minority classes (CBB, CBSD, CGM) automatically receive the heavy
    augmentation pipeline when augment=True; all other samples use the
    standard (or no-augmentation) pipeline.

    Args:
        dataframe  : DataFrame with columns 'image_id' and 'label'.
        images_dir : Directory containing raw image files.
        augment    : Apply training augmentations when True.
        image_size : Resize all images to this height/width.
    """

    def __init__(self, dataframe: pd.DataFrame, images_dir: str | Path, image_size: int = 224, augment: bool = True,) -> None:
        self.df         = dataframe.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.augment    = augment

        # Pre-build both pipelines once to avoid rebuilding per sample.
        self._robust_tf   = get_robust_transforms(image_size)
        self._standard_tf = get_transforms(image_size, augment=augment)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row   = self.df.iloc[idx]
        label = int(row["label"])

        image = cv2.imread(str(self.images_dir / row["image_id"]))
        if image is None:
            raise FileNotFoundError(f"Could not read: {self.images_dir / row['image_id']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Route minority classes to the heavier pipeline during training.
        tf = self._robust_tf if (self.augment and label in MINORITY_CLASSES) else self._standard_tf
        image = tf(image=image)["image"]

        return image, torch.tensor(label, dtype=torch.long)


#  Class-weight helpers 

def compute_class_weights(dataframe: pd.DataFrame, num_classes: int = 5) -> torch.Tensor:
    """
    Inverse-frequency class weights for CrossEntropyLoss / FocalLoss.
    Formula: w_c = N / (num_classes × count_c)
    CMD (~13 k) → ~0.17  |  CBB (~1.1 k) → ~2.0
    """
    counts = torch.zeros(num_classes)
    for label in dataframe["label"]:
        counts[int(label)] += 1
    counts = counts.clamp(min=1.0)
    return len(dataframe) / (num_classes * counts)


def build_weighted_sampler(dataframe: pd.DataFrame, num_classes: int = 5) -> WeightedRandomSampler:
    """
    WeightedRandomSampler so every batch has a balanced class mix.
    Pass as sampler= to DataLoader (instead of shuffle=True).
    """
    class_weights  = compute_class_weights(dataframe, num_classes)
    sample_weights = torch.tensor(
        [class_weights[int(lbl)] for lbl in dataframe["label"]],
        dtype=torch.float,
    )
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


#  Per-class F1 report 

def per_class_f1_report(all_labels: list[int], all_preds:  list[int], num_classes: int = 5,) -> pd.DataFrame:
    """
    DataFrame with precision, recall, F1 per class.
    Prefer this over overall accuracy — a CMD-biased model can hit 95%+
    accuracy while failing completely on CBB.
    """
    labels_t = torch.tensor(all_labels)
    preds_t  = torch.tensor(all_preds)
    rows = []
    for c in range(num_classes):
        tp = ((preds_t == c) & (labels_t == c)).sum().item()
        fp = ((preds_t == c) & (labels_t != c)).sum().item()
        fn = ((preds_t != c) & (labels_t == c)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        rows.append({
            "class":     CLASS_NAMES[c],
            "precision": round(precision, 3),
            "recall":    round(recall,    3),
            "f1":        round(f1,        3),
            "support":   int((labels_t == c).sum().item()),
        })
    return pd.DataFrame(rows)