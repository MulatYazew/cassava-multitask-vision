# AgroVision Africa — Cassava Leaf Disease Classification

Image classification pipeline for the Cassava Leaf Disease dataset (5
classes), with outlier-aware data cleaning, class-imbalance handling,
transfer-learning backbones, and a Streamlit demo. Tuned for Apple
Silicon (MPS) but falls back to CUDA/CPU automatically.

## Project structure

```
.
├── codes/
│   ├── config.py          # all hyperparameters / paths (single source of truth)
│   ├── utils.py            # seeding, device detection, helpers
│   ├── outlier_handler.py  # 3-stage outlier/QA pipeline
│   ├── data_handler.py     # Dataset, augmentations, class weights, sampler, per-class F1
│   ├── model.py            # CassavaCNN / EfficientNetV2-S / Swin-Tiny factory
│   ├── train.py             # Trainer: training loop, early stopping, checkpointing
│   └── evaluate.py          # Evaluator: metrics, confusion matrix, classification report
├── notebooks/
│   └── AgroVision_Africa.ipynb   # main pipeline (run this end to end)
├── cassava-leaf-dataset/
│   ├── train.csv
│   ├── label_num_to_disease_map.json
│   └── train_images/
├── models/                  # saved checkpoints (best_model.pth)
├── results/                  # metrics, plots, outlier reports
├── app.py                    # Streamlit demo
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the Kaggle Cassava Leaf Disease dataset under
`cassava-leaf-dataset/` (with `train.csv`, `train_images/`, and
`label_num_to_disease_map.json`).

## Reproducibility

- Global seed: `SEED = 42` (set in `codes/config.py`, applied via
  `utils.set_seed`).
- All hyperparameters (batch size, learning rate, epochs, patience,
  architecture choice, etc.) are centralized in `codes/config.py`.
- Device selection is automatic via `utils.get_device()`
  (MPS → CUDA → CPU).

## Running the pipeline

1. Open `notebooks/AgroVision_Africa.ipynb` and run cells top to bottom.
2. Steps performed:
   - Load and explore the dataset, visualize class distribution.
   - 3-stage outlier detection (`codes/outlier_handler.py`): file
     integrity, "not leaf-like" green-content check, and per-class
     embedding outliers (ResNet-50 features + IsolationForest).
   - Stratified 80/10/10 train/val/test split.
   - DataLoaders with albumentations augmentation (heavier pipeline for
     minority classes) and a class-weighted sampler.
   - Model selection (CassavaCNN / EfficientNet-V2-S / Swin-Tiny).
   - Training with weighted CrossEntropyLoss, AdamW, cosine LR
     annealing, and early stopping on validation loss; best checkpoint
     saved to `models/best_model.pth`.
   - Final evaluation **only on the best checkpoint**, on the held-out
     test set: accuracy/precision/recall/F1, per-class F1 report,
     classification report, and (raw + normalized) confusion matrices.
   - Optional: train and compare all three backbones.

## Demo

```bash
streamlit run app.py
```

Upload a cassava leaf photo to get a predicted disease class with
per-class confidence scores, using the checkpoint saved in `models/`.

## Disease classes

| ID | Class |
|----|-------|
| 0 | Cassava Bacterial Blight (CBB) |
| 1 | Cassava Brown Streak Disease (CBSD) |
| 2 | Cassava Green Mottle (CGM) |
| 3 | Cassava Mosaic Disease (CMD) |
| 4 | Healthy |
