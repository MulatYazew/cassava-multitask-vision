"""
train_isolation_forest.py
==========================
One-time script that trains and saves the Cassava Image Validation pipeline:
    ResNet-50  →  StandardScaler  →  PCA(64)  →  IsolationForest

The saved bundle is loaded by demo/app.py at inference time to gate incoming
images before disease classification runs.

Run from the project root:
    python codes/train_isolation_forest.py

Output:
    models/isolation_forest.pkl   — dict with keys: scaler, pca, iso

Reuses the same ResNet-50 transform and architecture defined in
codes/outlier_handler.py (LeafDataset.TFM + detect_embedding_outliers backbone).
"""

from __future__ import annotations

import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm
import joblib

from codes.config import DATA_DIR, IMAGE_DIR, MODELS_DIR

# ── Paths ─────────────────────────────────────────────────────────────────────
CSV_PATH   = DATA_DIR / "train.csv"
IMG_DIR    = IMAGE_DIR
OUTPUT_PKL = MODELS_DIR / "isolation_forest.pkl"

# ── Feature extraction settings (must match outlier_handler.LeafDataset.TFM) ─
BATCH_SIZE   = 64
NUM_WORKERS  = 0        # 0 is safest on macOS with MPS
PCA_DIMS     = 64
# IsolationForest contamination: fraction of images we expect to be outliers
# in the *training* set. 0.05 = 5%. Adjust if the dataset has known noise.
CONTAMINATION = 0.05

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Dataset ───────────────────────────────────────────────────────────────────

class CassavaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: Path):
        self.paths = [img_dir / str(r["image_id"]) for _, r in df.iterrows()]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        try:
            img = Image.open(self.paths[i]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        return TRANSFORM(img)


# ── Feature extraction ────────────────────────────────────────────────────────

def _get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def extract_features(df: pd.DataFrame, img_dir: Path, device: str) -> np.ndarray:
    """Extract 2048-dim ResNet-50 features for every image in df."""
    backbone = models.resnet50(weights="IMAGENET1K_V2")
    backbone.fc = nn.Identity()   # remove classification head → 2048-dim output
    backbone.eval().to(device)

    ds = CassavaDataset(df, img_dir)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    feats: list[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(dl, desc="  Extracting ResNet-50 features"):
            out = backbone(batch.to(device))
            feats.append(out.cpu().numpy())

    return np.vstack(feats)   # (N, 2048)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    device = _get_device()
    print(f"\nCassava Isolation Forest Trainer")
    print(f"{'─'*50}")
    print(f"  Device   : {device}")
    print(f"  CSV      : {CSV_PATH}")
    print(f"  Images   : {IMG_DIR}")
    print(f"  Output   : {OUTPUT_PKL}")
    print(f"{'─'*50}\n")

    # ── Load CSV ──────────────────────────────────────────────────────────────
    df = pd.read_csv(CSV_PATH)
    # Normalise column names (handles minor variations in the CSV header)
    col_map: dict[str, str] = {}
    for col in df.columns:
        if col.lower() in ("image_id", "img_id", "image", "filename", "id"):
            col_map[col] = "image_id"
        elif col.lower() in ("label", "class", "target", "disease"):
            col_map[col] = "label"
    df = df.rename(columns=col_map)
    if "image_id" not in df.columns:
        raise ValueError(f"Cannot find image_id column. Found: {df.columns.tolist()}")
    print(f"[1/4] Loaded CSV — {len(df):,} images across "
          f"{df['label'].nunique() if 'label' in df.columns else '?'} classes\n")

    # ── Extract ResNet-50 features ────────────────────────────────────────────
    print("[2/4] Extracting ResNet-50 features (2048 dims)…")
    feats = extract_features(df, IMG_DIR, device)
    print(f"      Feature matrix: {feats.shape}\n")

    # ── StandardScaler + PCA ─────────────────────────────────────────────────
    print("[3/4] Fitting StandardScaler + PCA…")
    scaler = StandardScaler()
    feats_scaled = scaler.fit_transform(feats)

    pca = PCA(n_components=PCA_DIMS, random_state=42)
    feats_pca = pca.fit_transform(feats_scaled)
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"      PCA: {PCA_DIMS} components, {var_explained:.1%} variance explained\n")

    # ── IsolationForest ───────────────────────────────────────────────────────
    print("[4/4] Fitting IsolationForest…")
    iso = IsolationForest(
        contamination=CONTAMINATION,
        random_state=42,
        n_jobs=-1,
        n_estimators=200,
    )
    iso.fit(feats_pca)

    # Sanity check: report score distribution on training data
    scores = iso.decision_function(feats_pca)
    print(f"      Score stats (training):")
    print(f"        min={scores.min():.4f}  max={scores.max():.4f}"
          f"  mean={scores.mean():.4f}  median={np.median(scores):.4f}")

    # ── Save bundle ───────────────────────────────────────────────────────────
    bundle = {
        "scaler": scaler,
        "pca":    pca,
        "iso":    iso,
        # Metadata for debugging / version tracking
        "meta": {
            "n_training_images": len(df),
            "pca_dims":          PCA_DIMS,
            "contamination":     CONTAMINATION,
            "score_mean":        float(scores.mean()),
            "score_median":      float(np.median(scores)),
        },
    }
    OUTPUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, OUTPUT_PKL)

    print(f"\n{'─'*50}")
    print(f"  Saved → {OUTPUT_PKL}")
    print(f"{'─'*50}")
    print("\nNext: run `streamlit run demo/app.py` — the validation layer will")
    print("load this model automatically.\n")


if __name__ == "__main__":
    main()
