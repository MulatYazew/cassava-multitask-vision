"""
cassava_outlier_handler.py
===========================
Outlier detection pipeline for the Cassava Leaf Disease dataset
(5 classes: cbb, cbsd, cgm, cmd, healthy).

DETECTION TARGETS
------------------
Stage 1 — File integrity & blank/corrupt images
    missing / truncated / corrupt / too small / near-black / near-white /
    low-contrast (solid-colour, scanner artefacts, failed captures).

Stage 2 — "Not leaf-like" detector
    Cassava photos should contain a meaningful amount of GREEN vegetation
    (leaf, stem, background foliage). Images that are almost entirely
    non-green (e.g. close-ups of soil, sky, hands, paper, or accidental
    photos of unrelated objects) are flagged.
    Uses: green_fraction (HSV hue in green range) + overall saturation.

Stage 3 — Per-class embedding outliers
    ResNet-50 features + per-class IsolationForest. Catches images that
    are visually very different from the rest of their disease class
    (wrong crop entirely, extreme blur, mislabeled image, etc.)

Usage
-----
    from cassava_outlier_handler import run_outlier_pipeline, visualize_flagged_images

    df, stats_df, flagged2, feats, ids, scores, flagged3 = run_outlier_pipeline(
        csv_path="train.csv", img_dir="train_images/", device="cuda"
    )

    visualize_flagged_images(flagged2, "train_images/", title="Stage 2 - not leaf-like")
    visualize_flagged_images(flagged3, "train_images/", title="Stage 3 - embedding outliers")

    # After visually confirming which ones are real outliers, build a
    # remove-list (list of image_ids) and call:
    final_df = apply_removals(df, remove_ids)
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm
from typing import Optional

try:
    from .config import RESULTS_DIR, DATA_DIR, IMAGE_DIR, MODELS_DIR
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from codes.config import RESULTS_DIR, DATA_DIR, IMAGE_DIR, MODELS_DIR

RESULTS_DIR = str(RESULTS_DIR)

BG, PANEL, ACCENT, FLAG, LINE, SPINE = (
    "#1a1a2e", "#12122a", "#7b7baa", "#ff4444", "#ffcc00", "#444466",
)


# STAGE 1 — File integrity / blank audit

def image_integrity_audit(
    df: pd.DataFrame,
    img_dir: str,
    min_bytes: int = 1_500,
    min_size_px: int = 64,
    min_std: float = 8.0,
    black_thresh: float = 12.0,
    white_thresh: float = 245.0,
) -> tuple:
    """Flag/remove only certain non-content images: missing, truncated,
    corrupt, too-small, near-black, near-white, or zero-variation."""
    issues = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="[Stage 1] Integrity audit"):
        path = os.path.join(img_dir, str(row["image_id"]))
        reason = None
        if not os.path.exists(path):
            reason = "missing"
        elif os.path.getsize(path) < min_bytes:
            reason = f"truncated(bytes={os.path.getsize(path)})"
        else:
            try:
                img = Image.open(path).convert("RGB")
                w, h = img.size
                if w < min_size_px or h < min_size_px:
                    reason = f"too_small({w}x{h})"
                else:
                    arr = np.array(img, dtype=np.float32)
                    mean, std = arr.mean(), arr.std()
                    if mean < black_thresh:
                        reason = f"near_black(mean={mean:.1f})"
                    elif mean > white_thresh:
                        reason = f"near_white(mean={mean:.1f})"
                    elif std < min_std:
                        reason = f"low_contrast(mean={mean:.0f},std={std:.1f})"
            except Exception as ex:
                reason = f"corrupt:{ex}"

        if reason:
            issues.append({"image_id": row["image_id"], "label": row["label"], "reason": reason})

    issues_df = pd.DataFrame(issues) if issues else pd.DataFrame(columns=["image_id", "label", "reason"])
    clean_df = df[~df["image_id"].isin(issues_df["image_id"])].reset_index(drop=True)

    print(f"\n[Stage 1] Auto-removed {len(issues_df):,} / {len(df):,} images")
    if not issues_df.empty:
        print(issues_df["reason"].str.split("(").str[0].value_counts().to_string())
        issues_df.to_csv(os.path.join(RESULTS_DIR, "removed_stage1_integrity.csv"), index=False)
        print("  -> saved: results/removed_stage1_integrity.csv")

    return clean_df, issues_df



# STAGE 2 — "Not leaf-like" detector (green-content check)

# A genuine cassava leaf photo (even a diseased/yellowing/spotted leaf,
# or a leaf shot against soil/sky) is expected to have SOME meaningful
# green/vegetation content and not be near-uniform in colour.
GREEN_HUE_RANGE = (35, 95)   # PIL HSV hue range (0-255 scale) covering green/yellow-green
MIN_GREEN_FRACTION = 0.05    # at least 5% of pixels look green/vegetation-like
MIN_SATURATION = 8.0         # near-zero saturation = greyscale / washed out photo


def leaf_stats(path: str) -> Optional[dict]:
    try:
        img = Image.open(path).convert("RGB")
        hsv = np.array(img.convert("HSV"), dtype=np.float32)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        green_mask = (h >= GREEN_HUE_RANGE[0]) & (h <= GREEN_HUE_RANGE[1]) & (s > 20)
        green_fraction = float(green_mask.mean())

        return {
            "image_id": None,
            "green_fraction": green_fraction,
            "saturation_mean": float(s.mean()),
            "brightness_mean": float(v.mean()),
            "aspect": img.width / img.height,
        }
    except Exception:
        return None


def detect_nonleaf_outliers(df: pd.DataFrame, img_dir: str) -> tuple:
    """Flag images with little/no green vegetation content or near-zero
    saturation (likely not a leaf photo, or a degraded/blank capture)."""
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="[Stage 2] Leaf-content stats"):
        path = os.path.join(img_dir, str(row["image_id"]))
        stats = leaf_stats(path)
        if stats is None:
            continue
        stats["image_id"] = row["image_id"]
        stats["label"] = row["label"]
        rows.append(stats)

    stats_df = pd.DataFrame(rows)
    if stats_df.empty:
        return stats_df, stats_df.copy()

    fail_mask = (stats_df["green_fraction"] < MIN_GREEN_FRACTION) | (stats_df["saturation_mean"] < MIN_SATURATION)
    reasons = []
    for _, r in stats_df.iterrows():
        rs = []
        if r["green_fraction"] < MIN_GREEN_FRACTION:
            rs.append(f"low_green={r['green_fraction']:.3f}")
        if r["saturation_mean"] < MIN_SATURATION:
            rs.append(f"low_saturation={r['saturation_mean']:.1f}")
        reasons.append("; ".join(rs))
    stats_df["fail_reasons"] = reasons

    flagged_df = stats_df[fail_mask].copy().reset_index(drop=True)

    print(f"\n[Stage 2] Flagged {len(flagged_df):,} / {len(stats_df):,} images as not leaf-like")
    if not flagged_df.empty:
        flagged_df.to_csv(os.path.join(RESULTS_DIR, "review_stage2_nonleaf.csv"), index=False)
        print("  -> saved: results/review_stage2_nonleaf.csv")

    return stats_df, flagged_df


# STAGE 3 — Per-class embedding outliers (ResNet-50 + IsolationForest)

class LeafDataset(Dataset):
    TFM = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def __init__(self, df: pd.DataFrame, img_dir: str):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        path = os.path.join(self.img_dir, str(row["image_id"]))
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        return self.TFM(img), int(row["label"]), str(row["image_id"])


def detect_embedding_outliers(
    df: pd.DataFrame,
    img_dir: str,
    device: str = "cpu",
    batch_size: int = 64,
    num_workers: int = 0,
    per_class_thr: float = -0.05,
    min_class_size: int = 30,
) -> tuple:
    """Extract ResNet-50 features and fit a per-class IsolationForest
    (cassava has only 5 classes, so per-class models are cheap and far
    more sensitive than a single global model)."""
    backbone = models.resnet50(weights="IMAGENET1K_V2")
    backbone.fc = nn.Identity()
    backbone.eval().to(device)

    ds = LeafDataset(df, img_dir)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=(device == "cuda"))

    feats_list, ids, labels = [], [], []
    with torch.no_grad():
        for imgs, lbls, img_ids in tqdm(dl, desc="[Stage 3] Feature extraction"):
            f = backbone(imgs.to(device)).cpu().numpy()
            feats_list.append(f)
            ids.extend(img_ids)
            labels.extend(lbls.tolist())

    feats = np.vstack(feats_list)
    labels = np.array(labels)

    feats_scaled = StandardScaler().fit_transform(feats)
    feats_pca = PCA(n_components=min(64, feats_scaled.shape[1]), random_state=42).fit_transform(feats_scaled)

    scores = np.zeros(len(ids))
    for lbl in np.unique(labels):
        mask = labels == lbl
        if mask.sum() < min_class_size:
            continue
        clf = IsolationForest(contamination="auto", random_state=42, n_jobs=-1)
        clf.fit(feats_pca[mask])
        scores[mask] = clf.decision_function(feats_pca[mask])

    outlier_mask = scores < per_class_thr
    outlier_ids = [ids[i] for i in np.where(outlier_mask)[0]]

    outlier_df = df[df["image_id"].astype(str).isin(outlier_ids)].copy()
    score_map = {ids[i]: float(scores[i]) for i in np.where(outlier_mask)[0]}
    outlier_df["anomaly_score"] = outlier_df["image_id"].astype(str).map(score_map)
    outlier_df = outlier_df.sort_values("anomaly_score").reset_index(drop=True)

    np.save(os.path.join(RESULTS_DIR, "embeddings_feats.npy"), feats_pca)
    np.save(os.path.join(RESULTS_DIR, "embeddings_scores.npy"), scores)
    with open(os.path.join(RESULTS_DIR, "embeddings_image_ids.txt"), "w") as f:
        f.write("\n".join(ids))

    print(f"\n[Stage 3] Flagged {len(outlier_df):,} / {len(df):,} images "
          f"(per-class IsolationForest, thr={per_class_thr})")
    if not outlier_df.empty:
        outlier_df.to_csv(os.path.join(RESULTS_DIR, "review_stage3_embedding.csv"), index=False)
        print("  -> saved: results/review_stage3_embedding.csv")

    return feats_pca, ids, scores, outlier_df


# VALIDATION MODEL TRAINING

def train_validation_model(
    csv_path: str,
    img_dir: str,
    output_pkl: str,
    device: str = "cpu",
    batch_size: int = 64,
    num_workers: int = 0,
    pca_dims: int = 64,
    contamination: float = 0.05,
) -> None:
    """Train and save the cassava image validation bundle.

    Fits a global ResNet-50 → StandardScaler → PCA → IsolationForest pipeline
    on all training images. The saved bundle is used by demo/app.py to gate
    uploaded images before disease classification runs.

    Run once from the project root:
        python codes/outlier_handler.py

    Output keys: scaler, pca, iso, meta
    """
    print(f"\nCassava Image Validation Trainer")
    print(f"{'─'*50}")
    print(f"  Device : {device}")
    print(f"  CSV    : {csv_path}")
    print(f"  Images : {img_dir}")
    print(f"  Output : {output_pkl}")
    print(f"{'─'*50}\n")
    # ── Load CSV ──────────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    col_map: dict = {}
    for col in df.columns:
        if col.lower() in ("image_id", "img_id", "image", "filename", "id"):
            col_map[col] = "image_id"
        elif col.lower() in ("label", "class", "target", "disease"):
            col_map[col] = "label"
    df = df.rename(columns=col_map)
    if "image_id" not in df.columns:
        raise ValueError(f"Cannot find image_id column. Found: {df.columns.tolist()}")
    if "label" not in df.columns:
        df["label"] = 0
    print(f"[1/4] Loaded {len(df):,} images across "
          f"{df['label'].nunique()} classes\n")
    # ── Extract ResNet-50 features (reuses LeafDataset.TFM) ──────────────────
    backbone = models.resnet50(weights="IMAGENET1K_V2")
    backbone.fc = nn.Identity()
    backbone.eval().to(device)

    ds = LeafDataset(df, img_dir)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=(device == "cuda"))

    print("[2/4] Extracting ResNet-50 features (2048 dims)…")
    feats_list: list = []
    with torch.no_grad():
        for imgs, _, _ in tqdm(dl, desc="  ResNet-50"):
            feats_list.append(backbone(imgs.to(device)).cpu().numpy())
    feats = np.vstack(feats_list)
    print(f"      Feature matrix: {feats.shape}\n")
    # ── StandardScaler + PCA ─────────────────────────────────────────────────
    print("[3/4] Fitting StandardScaler + PCA…")
    scaler = StandardScaler()
    feats_scaled = scaler.fit_transform(feats)
    pca = PCA(n_components=pca_dims, random_state=42)
    feats_pca = pca.fit_transform(feats_scaled)
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"      PCA: {pca_dims} components, {var_explained:.1%} variance explained\n")
    # ── IsolationForest ───────────────────────────────────────────────────────
    print("[4/4] Fitting IsolationForest…")
    iso = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
        n_estimators=200,
    )
    iso.fit(feats_pca)
    scores = iso.decision_function(feats_pca)
    print(f"      Score stats (training):")
    print(f"        min={scores.min():.4f}  max={scores.max():.4f}"
          f"  mean={scores.mean():.4f}  median={np.median(scores):.4f}")
    # ── Save bundle ───────────────────────────────────────────────────────────
    bundle = {
        "scaler": scaler,
        "pca":    pca,
        "iso":    iso,
        "meta": {
            "n_training_images": len(df),
            "pca_dims":          pca_dims,
            "contamination":     contamination,
            "score_mean":        float(scores.mean()),
            "score_median":      float(np.median(scores)),
        },
    }
    Path(output_pkl).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_pkl)

    print(f"\n{'─'*50}")
    print(f"  Saved → {output_pkl}")
    print(f"{'─'*50}")
    print("\nNext: streamlit run demo/app.py\n")


# VISUALISATION

def visualize_flagged_images(
    flagged_df: pd.DataFrame,
    img_dir: str,
    title: str = "Flagged images",
    max_images: int = 40,
    cols: int = 8,
    save_path: Optional[str] = None,
) -> None:
    """Grid view of flagged images with their reason / anomaly score.
    Inspect this to judge whether flags are real outliers (off-topic
    photos, blanks) or just unusual-but-valid leaf shots."""
    if flagged_df.empty:
        print("Nothing to plot - dataframe is empty.")
        return

    subset = flagged_df.head(max_images)
    n = len(subset)
    rows = max(1, (n + cols - 1) // cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.3, rows * 2.6), facecolor=BG)
    fig.suptitle(f"{title}  ({n} of {len(flagged_df)} shown)", fontsize=12, color="white", fontweight="bold", y=1.02)
    axes = np.array(axes).reshape(-1)

    for i, (_, row) in enumerate(subset.iterrows()):
        ax = axes[i]
        ax.set_facecolor(PANEL)
        path = os.path.join(img_dir, str(row["image_id"]))
        try:
            img = Image.open(path).convert("RGB")
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, "load error", ha="center", va="center", color="salmon", fontsize=7, transform=ax.transAxes)

        reason = ""
        for col in ("fail_reasons", "anomaly_score"):
            if col in row.index and pd.notna(row[col]) and str(row[col]) != "":
                val = row[col]
                reason = f"score={val:.4f}" if col == "anomaly_score" else str(val)[:50]
                break

        for sp in ax.spines.values():
            sp.set_edgecolor(FLAG); sp.set_linewidth(2)

        ax.set_title(f"cls {row['label']}\n{reason}", fontsize=5.5, color="white", pad=3)
        ax.axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off"); axes[j].set_facecolor(BG)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=120, facecolor=fig.get_facecolor())
        print(f"  -> saved grid: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_green_fraction_distribution(stats_df: pd.DataFrame, flagged_df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """Histogram of green_fraction across the dataset, with the Stage-2
    threshold marked. Use this to sanity-check / retune MIN_GREEN_FRACTION."""
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.hist(stats_df["green_fraction"], bins=100, color="#5555bb", alpha=0.85)
    ax.axvline(MIN_GREEN_FRACTION, color=FLAG, lw=2, ls="--", label=f"threshold={MIN_GREEN_FRACTION}")
    ax.set_xlabel("green_fraction", color="white")
    ax.set_ylabel("count", color="white")
    ax.set_title(f"Green-fraction distribution ({len(flagged_df)} flagged)", color="white", fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines[:].set_color(SPINE)
    ax.legend(labelcolor="white", facecolor=BG, edgecolor=SPINE)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=120, facecolor=fig.get_facecolor())
    else:
        plt.show()
    plt.close()


def plot_anomaly_score_distribution(scores: np.ndarray, thr: float = -0.15, save_path: Optional[str] = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.hist(scores, bins=150, color="#5555bb", alpha=0.85)
    ax.axvline(thr, color=FLAG, lw=2, ls="--", label=f"threshold={thr}")
    flagged = (scores < thr).sum()
    ax.text(thr, ax.get_ylim()[1] * 0.85, f"{flagged:,} flagged", color=FLAG, ha="right", fontsize=10)
    ax.set_xlabel("Anomaly score (per-class IsolationForest)", color="white")
    ax.set_ylabel("count", color="white")
    ax.set_title("Stage 3 - anomaly score distribution", color="white", fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines[:].set_color(SPINE)
    ax.legend(labelcolor="white", facecolor=BG, edgecolor=SPINE)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=120, facecolor=fig.get_facecolor())
    else:
        plt.show()
    plt.close()


# Apply removals

def apply_removals(df: pd.DataFrame, remove_ids, output_csv: Optional[str] = None) -> pd.DataFrame:
    """remove_ids: an iterable of image_id values confirmed (after visual
    review) to be real outliers."""
    if output_csv is None:
        output_csv = os.path.join(RESULTS_DIR, "train_clean.csv")
    remove_ids = set(str(x) for x in remove_ids)
    before = len(df)
    final_df = df[~df["image_id"].astype(str).isin(remove_ids)].reset_index(drop=True)
    final_df.to_csv(output_csv, index=False)
    print(f"\n[apply_removals] Removed {before - len(final_df):,} images.")
    print(f"  Final dataset: {len(final_df):,} images")
    print(f"  -> saved: {output_csv}")
    return final_df


# Master pipeline

def run_outlier_pipeline(
    csv_path: str,
    img_dir: str,
    device: str = "cpu",
    per_class_thr: float = -0.15,
    skip_stage3: bool = False,
) -> tuple:
    df = pd.read_csv(csv_path)
    col_map = {}
    for col in df.columns:
        if col.lower() in ("image_id", "img_id", "image", "filename", "id"):
            col_map[col] = "image_id"
        elif col.lower() in ("label", "class", "target", "disease"):
            col_map[col] = "label"
    df = df.rename(columns=col_map)

    if "image_id" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Cannot find image_id/label columns. Found: {df.columns.tolist()}")

    print(f"\nLoaded: {len(df):,} images - {df['label'].nunique()} classes")
    print("-" * 60)

    # Stage 1
    df, _ = image_integrity_audit(df, img_dir)

    # Stage 2
    stats_df, flagged2 = detect_nonleaf_outliers(df, img_dir)

    # Stage 3
    if skip_stage3:
        print("\n[Stage 3] Skipped.")
        feats, ids, scores, flagged3 = np.array([]), [], np.array([]), pd.DataFrame()
    else:
        feats, ids, scores, flagged3 = detect_embedding_outliers(
            df, img_dir, device=device, per_class_thr=per_class_thr
        )

    print(f"\n{'-'*60}")
    print("PIPELINE COMPLETE")
    print(f"{'-'*60}")
    print(f"  After Stage 1      : {len(df):,}")
    print(f"  Stage 2 flagged    : {len(flagged2):,}  -> review_stage2_nonleaf.csv")
    if not skip_stage3:
        print(f"  Stage 3 flagged    : {len(flagged3):,}  -> review_stage3_embedding.csv")
    print()
    print("NEXT STEPS:")
    print("  1. visualize_flagged_images(flagged2, img_dir, title='Stage 2 - not leaf-like')")
    print("  2. visualize_flagged_images(flagged3, img_dir, title='Stage 3 - embedding outliers')")
    print("  3. plot_green_fraction_distribution(stats_df, flagged2)  # tune threshold if needed")
    if not skip_stage3:
        print("  4. plot_anomaly_score_distribution(scores, per_class_thr)")
    print("  5. Build remove_ids list from confirmed outliers, then:")
    print("     final_df = apply_removals(df, remove_ids)")
    print(f"{'-'*60}\n")

    return df, stats_df, flagged2, feats, ids, scores, flagged3


if __name__ == "__main__":
    def get_device() -> str:
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    train_validation_model(
        csv_path=str(DATA_DIR / "train.csv"),
        img_dir=str(IMAGE_DIR),
        output_pkl=str(MODELS_DIR / "isolation_forest.pkl"),
        device=get_device(),
    )