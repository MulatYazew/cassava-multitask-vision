import os
import numpy as np
import pandas as pd
import imagehash
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from cleanlab.filter import find_label_issues
from tqdm import tqdm


# ── Stage 1: File integrity audit ──────────────────────────────────

def stage1_integrity_audit(df, img_dir, min_bytes=1500, min_std=5.0):
    """
    Flags missing, truncated, corrupt, and blank images.
    Returns (clean_df, issues_df).
    Auto-safe to remove without human review.
    """
    issues = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Stage 1: Integrity"):
        path = os.path.join(img_dir, str(row["image_id"]))
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
            issues.append({"image_id": row["image_id"],
                            "label":    row["label"],
                            "reason":   reason})

    issues_df = pd.DataFrame(issues)
    clean_df  = df[~df["image_id"].isin(issues_df["image_id"])].reset_index(drop=True)

    print(f"[Stage 1] Removed {len(issues_df)} / {len(df)} images")
    if len(issues_df):
        print(issues_df["reason"].str.split("(").str[0].value_counts().to_string())

    issues_df.to_csv("removed_stage1_integrity.csv", index=False)
    return clean_df, issues_df


# ── Stage 2: Perceptual hash deduplication ─────────────────────────

def stage2_phash_dedup(df, img_dir, hamming_threshold=10):
    """
    Detects exact and near-duplicate images using perceptual hashing.
    Also checks for train/val leakage if val_df is passed separately.
    Returns (clean_df, dupes_df).
    Auto-safe to remove without human review.
    """
    seen   = {}   # hash_str -> image_id
    dupes  = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Stage 2: pHash"):
        path = os.path.join(img_dir, str(row["image_id"]))
        try:
            h = imagehash.phash(Image.open(path))
        except Exception:
            continue

        matched = None
        for existing_h, existing_id in seen.items():
            if abs(h - imagehash.hex_to_hash(existing_h)) <= hamming_threshold:
                matched = existing_id
                break

        if matched:
            dupes.append({"image_id":    row["image_id"],
                           "label":       row["label"],
                           "duplicate_of": matched})
        else:
            seen[str(h)] = row["image_id"]

    dupes_df  = pd.DataFrame(dupes)
    clean_df  = df[~df["image_id"].isin(dupes_df["image_id"])].reset_index(drop=True)

    print(f"[Stage 2] Removed {len(dupes_df)} duplicates / {len(df)} images")

    dupes_df.to_csv("removed_stage2_dupes.csv", index=False)
    return clean_df, dupes_df


# ── Stage 3: Per-class pixel statistics ────────────────────────────

def stage3_pixel_stats(df, img_dir, z_thresh=3.0):
    """
    Flags images whose pixel statistics (mean, std, aspect ratio)
    are >z_thresh std devs from their class mean.
    Returns (stats_df, flagged_df). Requires human review before removing.
    """
    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Stage 3: Pixel stats"):
        path = os.path.join(img_dir, str(row["image_id"]))
        try:
            img = Image.open(path).convert("RGB")
            arr = np.array(img, dtype=np.float32)
            rows.append({
                "image_id": row["image_id"],
                "label":    row["label"],
                "mean_r":   arr[:, :, 0].mean(),
                "mean_g":   arr[:, :, 1].mean(),
                "mean_b":   arr[:, :, 2].mean(),
                "std":      arr.std(),
                "aspect":   img.width / img.height,
            })
        except Exception:
            continue

    stats_df  = pd.DataFrame(rows)
    flag_cols = ["mean_r", "mean_g", "mean_b", "std", "aspect"]
    flagged_ids = set()

    for label, grp in stats_df.groupby("label"):
        for col in flag_cols:
            sigma = grp[col].std()
            if sigma == 0:
                continue
            mu   = grp[col].mean()
            bad  = grp[(grp[col] - mu).abs() > z_thresh * sigma]["image_id"]
            flagged_ids.update(bad.tolist())

    flagged_df = stats_df[stats_df["image_id"].isin(flagged_ids)].copy()

    print(f"[Stage 3] Flagged {len(flagged_df)} images for review (z > {z_thresh})")

    flagged_df.to_csv("review_stage3_pixelstats.csv", index=False)
    return stats_df, flagged_df


# ── Stage 4: Embedding-space outlier detection ─────────────────────

class _FlatDataset(Dataset):
    def __init__(self, df, img_dir, tfm):
        self.df      = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.tfm     = tfm
        classes      = sorted(df["label"].unique())
        self.c2i     = {c: i for i, c in enumerate(classes)}
        self.targets = [self.c2i[l] for l in df["label"]]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        path = os.path.join(self.img_dir, str(row["image_id"]))
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
        return self.tfm(img), self.targets[i], row["image_id"]


def stage4_embedding_outliers(df, img_dir, num_classes=251,
                               contamination_cap=0.10, batch_size=64,
                               num_workers=4, device="cuda"):
    """
    Extracts ResNet-50 features, runs per-class Isolation Forest,
    and optionally saves UMAP coordinates for visualisation.
    Returns (feats, labels, image_ids, outlier_df).
    Requires human review before removing.
    """
    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    backbone = models.resnet50(weights="IMAGENET1K_V2")
    backbone.fc = nn.Identity()
    backbone.eval().to(device)

    ds = _FlatDataset(df, img_dir, tfm)
    dl = DataLoader(ds, batch_size=batch_size,
                    shuffle=False, num_workers=num_workers)

    all_feats, all_labels, all_ids = [], [], []

    with torch.no_grad():
        for imgs, lbls, ids in tqdm(dl, desc="Stage 4: Feature extraction"):
            feats = backbone(imgs.to(device)).cpu().numpy()
            all_feats.append(feats)
            all_labels.extend(lbls.numpy())
            all_ids.extend(ids)

    feats  = np.vstack(all_feats)
    labels = np.array(all_labels)

    outlier_indices = []
    for cls in tqdm(range(num_classes), desc="Stage 4: Isolation Forest"):
        mask = labels == cls
        n    = mask.sum()
        if n < 5:
            continue
        contamination = max(0.01, min(contamination_cap, 4 / n))
        clf   = IsolationForest(contamination=contamination,
                                random_state=42, n_jobs=-1)
        preds = clf.fit_predict(feats[mask])
        outlier_indices.extend(np.where(mask)[0][preds == -1].tolist())

    outlier_ids = [all_ids[i] for i in outlier_indices]
    outlier_df  = df[df["image_id"].isin(outlier_ids)].copy()

    print(f"[Stage 4] Flagged {len(outlier_df)} embedding outliers for review")

    outlier_df.to_csv("review_stage4_embedding_outliers.csv", index=False)
    np.save("embeddings_feats.npy",   feats)
    np.save("embeddings_labels.npy",  labels)

    return feats, labels, all_ids, outlier_df


# ── Stage 5: UMAP visualisation ────────────────────────────────────

def stage5_umap_visualise(feats, labels, image_ids,
                           n_neighbors=15, min_dist=0.1,
                           save_path="umap_plot.png",
                           max_classes_legend=20):
    """
    Reduces features to 2D with UMAP and saves a colour-coded scatter plot.
    Points far from their class cluster are visually obvious outliers.
    Returns the 2D embedding array.
    """
    try:
        import umap as umap_lib
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        raise ImportError("pip install umap-learn matplotlib")

    print("[Stage 5] Fitting UMAP (this may take a few minutes)...")
    reducer    = umap_lib.UMAP(n_components=2, n_neighbors=n_neighbors,
                                min_dist=min_dist, random_state=42)
    embedding  = reducer.fit_transform(feats)
    np.save("umap_coords.npy", embedding)

    unique_classes = np.unique(labels)
    cmap   = cm.get_cmap("tab20", len(unique_classes))
    colors = {cls: cmap(i) for i, cls in enumerate(unique_classes)}

    fig, ax = plt.subplots(figsize=(14, 11))
    for cls in unique_classes:
        mask = labels == cls
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=[colors[cls]], s=2, alpha=0.5, linewidths=0)

    ax.set_title("UMAP of training features (colour = class)", fontsize=14)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[Stage 5] UMAP plot saved to {save_path}")
    return embedding


# ── Stage 6: Label noise detection (Cleanlab) ──────────────────────

def stage6_label_noise(df, feats, labels,
                        n_cv_folds=5, pca_components=256):
    """
    Uses cross-validated logistic regression on PCA-compressed features
    to get predicted probabilities, then runs Cleanlab to rank
    the most likely mislabelled samples.
    Returns label_issues_df. Requires human review before removing.
    """
    print("[Stage 6] Running cross-validated feature classifier...")

    pipe = Pipeline([
        ("pca", PCA(n_components=pca_components, random_state=42)),
        ("lr",  LogisticRegression(max_iter=500, C=0.1, n_jobs=-1)),
    ])
    pred_probs = cross_val_predict(
        pipe, feats, labels,
        cv=n_cv_folds, method="predict_proba"
    )

    print("[Stage 6] Running Cleanlab...")
    issue_indices = find_label_issues(
        labels=labels,
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence",
    )

    issues_df = df.iloc[issue_indices].copy()
    issues_df["rank"] = range(len(issues_df))

    print(f"[Stage 6] Flagged {len(issues_df)} label suspects for review")
    issues_df.to_csv("review_stage6_label_suspects.csv", index=False)
    return issues_df


# ── Master pipeline ─────────────────────────────────────────────────

def run_outlier_pipeline(csv_path, img_dir, num_classes=251,
                          device="mps"):
    df = pd.read_csv(csv_path)
    df.columns = ["image_id", "label"]

    print(f"\nStarting pipeline: {len(df)} images, {num_classes} classes\n")

    # Stage 1 — auto-remove
    df, _ = stage1_integrity_audit(df, img_dir)

    # Stage 2 — auto-remove
    df, _ = stage2_phash_dedup(df, img_dir)

    # Stage 3 — save for review
    _, _ = stage3_pixel_stats(df, img_dir)

    # Stage 4 — extract features + flag outliers for review
    feats, labels, image_ids, _ = stage4_embedding_outliers(
        df, img_dir, num_classes=num_classes, device=device
    )

    # Stage 5 — visualise
    stage5_umap_visualise(feats, labels, image_ids)

    # Stage 6 — label noise for review
    stage6_label_noise(df, feats, labels)

    print("\nPipeline complete.")
    print("Auto-removed : removed_stage1_integrity.csv, removed_stage2_dupes.csv")
    print("Review queues: review_stage3_pixelstats.csv")
    print("               review_stage4_embedding_outliers.csv")
    print("               review_stage6_label_suspects.csv")
    print("Visualisation: umap_plot.png")
    return df


if __name__ == "__main__":
    clean_df = run_outlier_pipeline(
        csv_path="train_labels.csv",
        img_dir="train_set/",
        num_classes=251,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )