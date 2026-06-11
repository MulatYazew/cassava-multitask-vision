"""
Dataset and DataLoader definitions for AgroVision project.
Handles data loading, preprocessing, and augmentation.
"""
import os
import time
import sys
import json
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, WeightedRandomSampler
import matplotlib.pyplot as plt
from tqdm import tqdm


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

# Classes with < 3 000 samples — receive heavier augmentation.
MINORITY_CLASSES: set[int] = {0, 1, 2}   # CBB (~1 100), CBSD (~2 200), CGM (~2 400)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)



# Label map helper

def load_label_map(dataset_dir: str | Path) -> dict[int, str]:
    """Load the official Kaggle label JSON if present, else fall back to CLASS_NAMES."""
    path = Path(dataset_dir) / "label_num_to_disease_map.json"
    if not path.exists():
        return CLASS_NAMES
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}

# Data Augmentation pipelines to handle class imbalance and improve generalisation to real-world field conditions.

def get_transforms(image_size: int = 224, augment: bool = True) -> A.Compose:
    """
    Standard augmentation pipeline.

    Used for majority classes during training and for validation / inference.
    Each transform is justified for cassava leaf classification:

    - RandomResizedCrop  : random scale + crop simulates varying distances from
                           the leaf and improves spatial invariance.
    - HorizontalFlip     : cassava leaves have no preferred left/right orientation.
    - Rotate             : field photos are rarely perfectly upright.
    - BrightnessContrast : lighting conditions vary significantly across Africa.
    - ColorJitter        : handles white-balance differences in cheap smartphones.
    - Affine             : small translations and scale changes add spatial diversity.
    - Normalize          : ImageNet statistics match the pretrained backbone priors.
    """
    if augment:
        return A.Compose([
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=25, border_mode=0, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.Affine(translate_percent=0.05, scale=(0.90, 1.10), rotate=(-15, 15), p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    # Validation / inference: deterministic resize + normalise only.
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_robust_transforms(image_size: int = 224) -> A.Compose:
    """
    Heavy augmentation pipeline for minority classes (CBB, CBSD, CGM).

    Extra operations vs the standard pipeline, each motivated by real-world
    African field conditions:

    - VerticalFlip       : cassava leaves have no canonical top/bottom orientation.
    - GaussianBlur       : mimics motion blur and low-quality smartphone optics.
    - ImageCompression   : simulates JPEG artefacts from cheap devices with small
                           internal storage that aggressively compress images.
    - CoarseDropout      : forces the model to use partial information, improving
                           robustness when lesions are partially obscured by dust,
                           fingers, or overlapping leaves.
    - RandomShadow       : simulates partial shadows cast by surrounding vegetation
                           — common in field photography under a canopy.
    """
    return A.Compose([
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.5, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.4),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=35, border_mode=0, p=0.6),
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.6),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.7),
        A.Affine(translate_percent=0.08, scale=(0.85, 1.15), rotate=(-20, 20), p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.4),
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

# Dataset

class CassavaDataset(Dataset):
    """
    PyTorch Dataset for cassava leaf images.

    Minority classes (CBB, CBSD, CGM) automatically receive the robust
    augmentation pipeline when ``augment=True``; majority classes and all
    validation samples use the standard (or no-augmentation) pipeline.

    Args:
        dataframe  : DataFrame with columns ``image_id`` and ``label``.
        images_dir : Root directory containing the raw image files.
        augment    : If True, apply training augmentations.
        image_size : Height/width to resize all images to.
    """

    def __init__( self, dataframe:  pd.DataFrame, images_dir: str | Path, augment:    bool = True, image_size: int  = 224,) -> None:
        self.df         = dataframe.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.augment    = augment
        self.image_size = image_size

        # Pre-build both pipelines once — avoids rebuilding per sample.
        self._robust_tf   = get_robust_transforms(image_size)
        self._standard_tf = get_transforms(image_size, augment=augment)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row   = self.df.iloc[idx]
        label = int(row["label"])

        image_path = self.images_dir / row["image_id"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Route minority classes to the heavier augmentation pipeline.
        if self.augment and label in MINORITY_CLASSES:
            image = self._robust_tf(image=image)["image"]
        else:
            image = self._standard_tf(image=image)["image"]

        return image, torch.tensor(label, dtype=torch.long)


# Class-weight helpers

def compute_class_weights(dataframe:   pd.DataFrame, num_classes: int = 5,) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from a training DataFrame.

    Returns a float tensor of shape (num_classes,) suitable for passing
    directly to ``CrossEntropyLoss(weight=...)`` or ``FocalLoss(alpha=...)``.

    Formula: w_c = N / (num_classes * count_c)
      Keeps weights proportional across classes.
      CMD (~13 k) → weight ~0.17 ;  CBB (~1.1 k) → weight ~2.0
    """
    counts = torch.zeros(num_classes)
    for label in dataframe["label"]:
        counts[int(label)] += 1
    counts  = counts.clamp(min=1.0)           # guard against absent classes
    weights = len(dataframe) / (num_classes * counts)
    return weights


def build_weighted_sampler(dataframe: pd.DataFrame, num_classes: int = 5,) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler so every training batch contains a
    balanced mix of all five classes regardless of raw class counts.

    Pass this as ``sampler=`` to DataLoader instead of ``shuffle=True``::

        loader = DataLoader(dataset, batch_size=32,
                            sampler=build_weighted_sampler(train_df))
    """
    class_weights  = compute_class_weights(dataframe, num_classes)
    sample_weights = torch.tensor(
        [class_weights[int(label)] for label in dataframe["label"]],
        dtype=torch.float,
    )
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,   # required: allows minority samples to repeat
    )



# Evaluation helper                                                          


def per_class_f1_report(all_labels: list[int], all_preds:  list[int], num_classes: int = 5,) -> pd.DataFrame:
    """
    Return a DataFrame with precision, recall, and F1 per class.

    Prefer this over overall accuracy — a model biased toward CMD will show
    95 %+ accuracy while completely failing on CBB.

    Usage::

        report = per_class_f1_report(true_labels, pred_labels)
        failing = report[report["f1"] < 0.80]
        if not failing.empty:
            print("Classes needing attention:", failing["class"].tolist())
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
        f1        = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        rows.append({
            "class":     CLASS_NAMES[c],
            "precision": round(precision, 3),
            "recall":    round(recall,    3),
            "f1":        round(f1,        3),
            "support":   int((labels_t == c).sum().item()),
        })

    return pd.DataFrame(rows)

def audit_images(df, img_dir, min_std=5.0, min_bytes=1500):
    """
    Flags missing, truncated, corrupt, and blank images.
    Returns (clean_df, issues_df).
    """
    issues = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Stage 1: Integrity"):
        path   = os.path.join(img_dir, str(row["image_id"]))
        reason = None

        if not os.path.exists(path):
            reason = "missing"
        elif os.path.getsize(path) < min_bytes:
            reason = "truncated"
        else:
            try:
                arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32)
                if arr.std() < min_std:
                    reason = f"blank(mean={arr.mean():.0f},std={arr.std():.1f})"
            except Exception as e:
                reason = f"corrupt:{e}"

        if reason:
            issues.append({
                "image_id": row["image_id"],
                "label":    row["label"],
                "reason":   reason
            })

    issues_df = pd.DataFrame(issues) if issues else pd.DataFrame(
        columns=["image_id", "label", "reason"]
    )
    clean_df  = df[~df["image_id"].isin(issues_df["image_id"])].reset_index(drop=True)

    print(f"[Stage 1] Removed {len(issues_df)} / {len(df)} images")
    if not issues_df.empty:
        print(issues_df["reason"].str.split("(").str[0].value_counts().to_string())

    issues_df.to_csv("removed_stage1_integrity.csv", index=False)
    return clean_df, issues_df



def show_outlier_samples(outliers_df, img_dir, img_col='image_id', class_col='label', samples_per_class=5):

    classes = sorted(outliers_df[class_col].unique())

    fig, axes = plt.subplots(
        len(classes),
        samples_per_class,
        figsize=(4*samples_per_class, 4*len(classes))
    )

    if len(classes) == 1:
        axes = [axes]

    for r, cls in enumerate(classes):

        cls_outliers = outliers_df[
            outliers_df[class_col] == cls
        ]

        sample_df = cls_outliers.sample(
            min(samples_per_class, len(cls_outliers)),
            random_state=42
        )

        for c in range(samples_per_class):

            ax = axes[r][c]

            if c >= len(sample_df):
                ax.axis("off")
                continue

            row = sample_df.iloc[c]

            path = os.path.join(
                img_dir,
                str(row[img_col])
            )

            try:
                img = Image.open(path).convert("RGB")

                ax.imshow(img)

                ax.set_title(
                    f"{cls}\nBrightness={row['brightness']:.2f}",
                    fontsize=9
                )

            except Exception as e:

                ax.text(
                    0.5,
                    0.5,
                    f"Error\n{e}",
                    ha='center'
                )

            ax.axis("off")

    plt.tight_layout()
    plt.show()
