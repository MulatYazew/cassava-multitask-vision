"""
Cassava Vision AI
=================
AI-Powered Crop Disease Detection & Explainable Diagnostics Platform

Supports three trained models:
  • EfficientNet Model  → models/efficientnet_v2_s_best_model.pth
  • ConvNeXt Model      → models/cassava_cnn_best_model.pth
  • Vision Transformer  → models/swin_tiny_best_model.pth

Run:
    streamlit run demo/app.py     (from project root)
    streamlit run app.py          (from inside demo/)
"""

from __future__ import annotations
# ── stdlib ────────────────────────────────────────────────────────────────────
import io
import re
import sys
import time
import zipfile
import datetime
from pathlib import Path
from typing import Optional
# ── core deps (always available) ──────────────────────────────────────────────
import numpy as np
import streamlit as st
from PIL import Image
# ── optional deps (fail gracefully) ──────────────────────────────────────────
try:
    import pandas as pd
    _PANDAS_OK = True
except ImportError:
    _PANDAS_OK = False

try:
    import plotly.graph_objects as go
    _PLOTLY_OK = True
except ImportError:
    _PLOTLY_OK = False

try:
    import cv2
    _CV2_OK = True
except ImportError:
    _CV2_OK = False

try:
    import matplotlib
    matplotlib.use("Agg")
    _MPL_OK = True
except ImportError:
    _MPL_OK = False

try:
    import requests as _req
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

try:
    import torch
    import torch.nn.functional as F
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

try:
    import torch.nn as _nn
    from torchvision import models as _tv_models, transforms as _tv_transforms
    _TORCHVISION_OK = True
except ImportError:
    _TORCHVISION_OK = False

try:
    import joblib
    _JOBLIB_OK = True
except ImportError:
    _JOBLIB_OK = False

try:
    __import__("openpyxl")
    _OPENPYXL_OK = True
except ImportError:
    import subprocess as _sp
    _sp.check_call([sys.executable, "-m", "pip", "install", "openpyxl", "-q"])
    _OPENPYXL_OK = True
# ── path / project imports ────────────────────────────────────────────────────
FILE     = Path(__file__).resolve()
DEMO_DIR = FILE.parent
ROOT     = DEMO_DIR
for _c in [DEMO_DIR, DEMO_DIR.parent]:
    if (_c / "codes").exists():
        ROOT = _c
        break
sys.path.insert(0, str(ROOT))

try:
    from codes.config import NUM_CLASSES, INPUT_SIZE, MODELS_DIR
    from codes.data_handler import CLASS_NAMES, CLASS_DESCRIPTIONS, get_transforms
    from codes.model import build_model
    from codes.utils import get_device
    from codes.gradcam import GradCAM, overlay_heatmap, preprocess_for_gradcam
    _CODES_OK = True
except Exception:
    _CODES_OK   = False
    NUM_CLASSES = 5
    INPUT_SIZE  = 224
    MODELS_DIR  = ROOT / "models"
    CLASS_NAMES = {
        0: "Cassava Bacterial Blight (CBB)",
        1: "Cassava Brown Streak Disease (CBSD)",
        2: "Cassava Green Mottle (CGM)",
        3: "Cassava Mosaic Disease (CMD)",
        4: "Healthy",
    }
    CLASS_DESCRIPTIONS = {
        0: "Bacterial infection causing angular water-soaked leaf spots, wilting and stem dieback.",
        1: "Viral disease producing yellow/brown streaks along leaf veins and brown root necrosis.",
        2: "Mild viral disease with mottled green patches and mosaic patterns on leaves.",
        3: "Severe viral disease causing mosaic patterns, leaf distortion and significant yield loss.",
        4: "No visible disease symptoms detected. Plant appears healthy and vigorous.",
    }

# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

DISEASE_CLASSES = {
    0: "Cassava Bacterial Blight (CBB)",
    1: "Cassava Brown Streak Disease (CBSD)",
    2: "Cassava Green Mottle (CGM)",
    3: "Cassava Mosaic Disease (CMD)",
    4: "Healthy",
}

CLASS_SHORT = {0: "CBB", 1: "CBSD", 2: "CGM", 3: "CMD", 4: "Healthy"}

SEVERITY       = {0: "High",   1: "High",   2: "Medium", 3: "High",  4: "None"}
SEVERITY_COLOR = {"High": "#C62828", "Medium": "#FF8F00", "None": "#2E7D32"}

TREATMENTS = {
    0: ("Remove and burn infected plant material immediately. "
        "Apply copper-based bactericide (Kocide 3000). "
        "Source certified disease-free cuttings for replanting."),
    1: ("Rogue out infected plants — no chemical cure exists for CBSD. "
        "Plant CBSD-tolerant varieties (Narocass 1, NASE 14). "
        "Control whitefly vectors with imidacloprid or neem spray."),
    2: ("Monitor closely — CGM is generally mild. "
        "Maintain healthy soil nutrition with balanced NPK. "
        "Apply neem-based spray to suppress mealybug vectors."),
    3: ("Uproot severely infected plants (>50% distortion). "
        "Plant CMD-resistant varieties (TMEB419, Serere). "
        "Eliminate whitefly vectors with neem oil fortnightly."),
    4: ("No action required. Continue scouting every 2 weeks. "
        "Maintain good soil health with compost and balanced fertiliser. "
        "Remove weeds that harbour whitefly vectors."),
}

RECOMMENDATION_STEPS = {
    0: [
        {"icon": "🔴", "step": "Isolate immediately",    "detail": "Quarantine affected plants to prevent spread."},
        {"icon": "🪣", "step": "Apply bactericide",      "detail": "Spray copper-based bactericide covering leaf undersides."},
        {"icon": "✂️",  "step": "Remove infected tissue", "detail": "Burn infected parts; sterilise tools between plants."},
        {"icon": "🌱", "step": "Replant clean cuttings", "detail": "Source certified disease-free cuttings from a nursery."},
    ],
    1: [
        {"icon": "🚨", "step": "Rogue out now",            "detail": "Uproot and burn — no chemical cure exists for CBSD."},
        {"icon": "🦟", "step": "Control whiteflies",       "detail": "Apply imidacloprid or neem spray to reduce vector load."},
        {"icon": "🌾", "step": "Plant tolerant varieties", "detail": "Restock with Narocass 1, NASE 14 or local certified stock."},
        {"icon": "🔍", "step": "Scout weekly",             "detail": "Inspect surrounding plants every 7 days."},
    ],
    2: [
        {"icon": "📋", "step": "Monitor closely",         "detail": "Tag the plant and check for symptom progression weekly."},
        {"icon": "🌿", "step": "Boost soil health",       "detail": "Apply balanced NPK and organic compost."},
        {"icon": "🌱", "step": "Use certified material",  "detail": "Source virus-tested cuttings for next season."},
        {"icon": "🦟", "step": "Reduce vector pressure",  "detail": "Apply neem spray to suppress mealybug populations."},
    ],
    3: [
        {"icon": "🚨", "step": "Uproot severe cases",      "detail": "Remove plants with >50% leaf distortion or stunting."},
        {"icon": "🌾", "step": "Plant resistant varieties", "detail": "Restock with TMEB419, Serere or CMD-resistant stock."},
        {"icon": "🦟", "step": "Eliminate whiteflies",     "detail": "Spray neem oil or imidacloprid fortnightly."},
        {"icon": "🔍", "step": "Scout every 2 weeks",      "detail": "Use yellow sticky traps for early detection."},
    ],
    4: [
        {"icon": "✅", "step": "No treatment needed",  "detail": "Continue normal management practices."},
        {"icon": "🔍", "step": "Scout every 2 weeks",  "detail": "Routine inspections catch diseases before they spread."},
        {"icon": "🌿", "step": "Maintain soil health", "detail": "Apply compost and balanced fertiliser."},
        {"icon": "🌱", "step": "Manage weeds",         "detail": "Remove weeds that harbour vectors near cassava beds."},
    ],
}

CROPS = ["Cassava", "Maize", "Tomato", "Potato", "Rice"]

DISEASE_LINKS = {
    0: "https://en.wikipedia.org/wiki/Cassava_bacterial_blight",
    1: "https://en.wikipedia.org/wiki/Cassava_brown_streak_disease",
    2: "https://en.wikipedia.org/wiki/Cassava_green_mottle_virus",
    3: "https://en.wikipedia.org/wiki/African_cassava_mosaic_disease",
    4: "https://en.wikipedia.org/wiki/Cassava",
}

# Sidebar display name → internal model key → checkpoint filename
MODELS_META = {
    "EfficientNet-V2-S": {
        "key":   "efficientnet_v2_s",
        "ckpt":  "efficientnet_v2_s_best_model.pth",
        "tag":   "CNN · Recommended",
        "desc":  "EfficientNet-V2-S with Fused-MBConv blocks. Best accuracy-to-compute ratio. Recommended for field deployment.",
        "acc":   "92.4%",
        "params":"21.5M",
    },
    "CassavaCNN (Custom)": {
        "key":   "cassava_cnn",
        "ckpt":  "cassava_cnn_best_model.pth",
        "tag":   "CNN · From Scratch",
        "desc":  "Custom CNN built from scratch with 3×3 residual blocks and Squeeze-and-Excitation attention (~9.3M params). No pretrained weights.",
        "acc":   "88.7%",
        "params":"9.3M",
    },
    "Swin-Tiny + LoRA": {
        "key":   "swin_tiny",
        "ckpt":  "swin_tiny_best_model.pth",
        "tag":   "ViT · LoRA rank=8",
        "desc":  "Swin Transformer Tiny fine-tuned with LoRA (rank=8, α=16) on qkv & proj layers. Best explainability.",
        "acc":   "91.8%",
        "params":"28.3M",
    },
}

VALID_EXTS   = {".jpg", ".jpeg", ".png", ".webp"}
MAX_ZIP_MB   = 500
MAX_HISTORY  = 5

# OOD thresholds (applied only when a real model is loaded)
OOD_MAX_PROB  = 0.22   # below this → likely not a cassava leaf
OOD_ENTROPY   = 0.92   # normalised entropy above this → model is very confused
# ── Isolation Forest gate threshold ───────────────────────────────────────────
# iso.decision_function() returns negative scores for outliers.
# Scores below OOD_THRESHOLD → image rejected as non-cassava.
# Tune: more negative = more permissive (fewer rejections).
# Run codes/outlier_handler.py once to generate models/isolation_forest.pkl.
OOD_THRESHOLD = -0.05

# ═════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Cassava Vision AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

/* ── design tokens ── */
:root {
  --primary:    #2E7D32;
  --primary-lt: #43A047;
  --primary-bg: #E8F5E9;
  --primary-bd: #C8E6C9;
  --accent:     #FF8F00;
  --accent-bg:  #FFF8E1;
  --danger:     #C62828;
  --danger-bg:  #FFEBEE;
  --warn:       #F57F17;
  --warn-bg:    #FFF9C4;
  --info:       #1565C0;
  --info-bg:    #E3F2FD;
  --bg:         #F5F7FA;
  --surface:    #FFFFFF;
  --surface-2:  #FAFAFA;
  --border:     #E0E0E0;
  --border-2:   #BDBDBD;
  --text:       #263238;
  --text-2:     #546E7A;
  --text-3:     #90A4AE;
  --radius:     16px;
  --radius-sm:  10px;
  --radius-xs:  6px;
  --shadow:     0 2px 12px rgba(0,0,0,0.08);
  --shadow-md:  0 4px 24px rgba(0,0,0,0.12);
  --shadow-lg:  0 8px 40px rgba(0,0,0,0.16);
  --mono:       'JetBrains Mono', monospace;
  --sans:       'Inter', sans-serif;
  --ease:       all 0.2s cubic-bezier(0.4,0,0.2,1);
}

/* ── reset ── */
.stApp { background: var(--bg) !important; font-family: var(--sans); color: var(--text); }
* { box-sizing: border-box; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 4rem !important; max-width: 1400px !important; }
h1,h2,h3,h4 { font-family: var(--sans); letter-spacing: -0.02em; color: var(--text); }

/* ── hero ── */
.hero {
  background: linear-gradient(135deg,#1B5E20 0%,#2E7D32 45%,#388E3C 75%,#43A047 100%);
  border-radius: 20px; padding: 2.25rem 3rem;
  margin-bottom: 1.5rem; position: relative; overflow: hidden;
  box-shadow: var(--shadow-lg);
}
.hero::before {
  content:''; position:absolute; top:-90px; right:-90px;
  width:380px; height:380px;
  background:radial-gradient(circle,rgba(255,255,255,.08) 0%,transparent 60%);
  border-radius:50%; pointer-events:none;
}
.hero::after {
  content:''; position:absolute; bottom:-60px; left:25%;
  width:260px; height:260px;
  background:radial-gradient(circle,rgba(255,143,0,.12) 0%,transparent 60%);
  border-radius:50%; pointer-events:none;
}
.hero-inner {
  position:relative; z-index:1;
  display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:1.5rem;
}
.hero-brand { display:flex; align-items:center; gap:1.25rem; }
.hero-logo {
  width:70px; height:70px; border-radius:18px; flex-shrink:0;
  background:rgba(255,255,255,.15); border:2px solid rgba(255,255,255,.25);
  display:flex; align-items:center; justify-content:center; font-size:34px;
  backdrop-filter:blur(8px); box-shadow:0 4px 20px rgba(0,0,0,.2);
}
.hero-title { font-size:1.95rem; font-weight:800; color:#fff; margin:0; line-height:1.1; }
.hero-sub   { font-size:.82rem; color:rgba(255,255,255,.72); margin-top:5px; }
.hero-stats { display:flex; gap:2rem; flex-wrap:wrap; }
.hero-stat-v { font-family:var(--mono); font-size:1.7rem; font-weight:700; color:#fff; line-height:1; }
.hero-stat-l { font-size:.66rem; color:rgba(255,255,255,.62); text-transform:uppercase; letter-spacing:.1em; margin-top:3px; }

/* ── stat strip ── */
.stat-strip {
  display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin-bottom:1.5rem;
}
.stat-card {
  background:var(--surface); border:1px solid var(--border); border-radius:var(--radius-sm);
  padding:1rem 1.25rem; box-shadow:var(--shadow);
  display:flex; align-items:center; gap:1rem; transition:var(--ease);
}
.stat-card:hover { box-shadow:var(--shadow-md); transform:translateY(-1px); }
.stat-icon {
  width:44px; height:44px; border-radius:12px; flex-shrink:0;
  display:flex; align-items:center; justify-content:center; font-size:22px;
}
.stat-icon.g { background:var(--primary-bg); }
.stat-icon.o { background:var(--accent-bg); }
.stat-icon.b { background:var(--info-bg); }
.stat-icon.r { background:var(--danger-bg); }
.stat-val { font-family:var(--mono); font-size:1.4rem; font-weight:700; color:var(--text); line-height:1; }
.stat-lbl { font-size:.68rem; color:var(--text-3); text-transform:uppercase; letter-spacing:.08em; margin-top:3px; }

/* ── section label ── */
.sec-label {
  font-size:.68rem; font-weight:700; text-transform:uppercase; letter-spacing:.14em;
  color:var(--text-3); margin:0 0 .85rem;
  display:flex; align-items:center; gap:8px;
}
.sec-num {
  background:var(--primary-bg); color:var(--primary);
  font-size:.6rem; width:22px; height:22px; border-radius:var(--radius-xs);
  display:flex; align-items:center; justify-content:center; font-weight:700; flex-shrink:0;
}

/* ── upload zone ── */
.upload-zone {
  border:2px dashed var(--border-2); border-radius:var(--radius);
  background:var(--surface); overflow:hidden; transition:var(--ease);
}
.upload-zone:hover { border-color:var(--primary); background:var(--primary-bg); }
.upload-zone-hdr { padding:1.75rem 1.5rem .5rem; text-align:center; pointer-events:none; }
.upload-icon { font-size:2.5rem; margin-bottom:.5rem; }
.upload-title { font-size:.95rem; font-weight:600; color:var(--text); margin-bottom:4px; }
.upload-hint  { font-size:.74rem; color:var(--text-3); font-family:var(--mono); }
.chip-row { display:flex; justify-content:center; gap:6px; margin-top:10px; flex-wrap:wrap; }
.chip {
  background:var(--primary-bg); color:var(--primary); border:1px solid var(--primary-bd);
  font-family:var(--mono); font-size:.6rem; font-weight:600;
  padding:3px 10px; border-radius:20px;
}

/* ── camera zone ── */
.camera-zone {
  border:2px solid #A5D6A7; border-radius:var(--radius);
  background:linear-gradient(135deg,#F1F8E9 0%,var(--surface) 100%); overflow:hidden;
}
.camera-hdr {
  padding:.8rem 1.25rem .5rem; border-bottom:1px solid #C8E6C9;
  display:flex; align-items:center; gap:10px;
}
.cam-pulse {
  width:9px; height:9px; border-radius:50%; background:var(--primary); flex-shrink:0;
  box-shadow:0 0 0 0 rgba(46,125,50,.5);
  animation:cam-p 2s infinite;
}
@keyframes cam-p {
  0%  { box-shadow:0 0 0 0 rgba(46,125,50,.5); }
  70% { box-shadow:0 0 0 8px rgba(46,125,50,0); }
  100%{ box-shadow:0 0 0 0 rgba(46,125,50,0); }
}
.camera-title { font-family:var(--mono); font-size:.75rem; font-weight:600; color:var(--primary); }
.camera-body  { padding:.75rem 1rem 1rem; }
div[data-testid="stCameraInput"] { background:transparent !important; border:none !important; }
div[data-testid="stCameraInput"] video { border-radius:var(--radius-sm) !important; width:100% !important; }

/* ── cards ── */
.cv-card {
  background:var(--surface); border:1px solid var(--border); border-radius:var(--radius);
  padding:1.5rem; box-shadow:var(--shadow); margin-bottom:1rem; transition:var(--ease);
}
.cv-card:hover { box-shadow:var(--shadow-md); }

/* ── prediction card ── */
.pred-card {
  background:var(--surface); border:1px solid var(--border);
  border-radius:var(--radius); padding:1.5rem;
  position:relative; overflow:hidden; box-shadow:var(--shadow);
}
.pred-card::before {
  content:''; position:absolute; top:0; left:0; right:0; height:4px;
}
.pred-card.high::before  { background:linear-gradient(90deg,var(--danger),#EF5350); }
.pred-card.med::before   { background:linear-gradient(90deg,var(--accent),#FFB300); }
.pred-card.none::before  { background:linear-gradient(90deg,var(--primary),var(--primary-lt)); }
.pred-eyebrow { font-size:.62rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; color:var(--text-3); margin-bottom:4px; font-family:var(--mono); }
.pred-name    { font-size:1.4rem; font-weight:700; color:var(--text); letter-spacing:-.02em; }
.pred-desc    { font-size:.82rem; color:var(--text-2); line-height:1.7; margin-top:.75rem; }
.conf-pill    { display:inline-flex; align-items:center; gap:6px; padding:6px 14px; border-radius:20px; font-family:var(--mono); font-size:.8rem; font-weight:700; }
.conf-dot     { width:7px; height:7px; border-radius:50%; }
.sev-badge    { display:inline-flex; align-items:center; gap:6px; padding:4px 10px; border-radius:20px; font-size:.72rem; font-weight:600; font-family:var(--mono); margin-top:6px; }

/* ── confidence bar ── */
.conf-wrap {
  background:var(--surface-2); border:1px solid var(--border);
  border-radius:var(--radius-sm); padding:1rem 1.25rem; margin:1rem 0;
}
.conf-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; }
.conf-lbl { font-size:.68rem; font-weight:700; text-transform:uppercase; letter-spacing:.1em; color:var(--text-3); }
.conf-val { font-family:var(--mono); font-size:1.25rem; font-weight:700; }
.conf-bar { width:100%; height:12px; background:var(--border); border-radius:8px; overflow:hidden; }
.conf-fill {
  height:100%; border-radius:8px; transition:width .8s cubic-bezier(.4,0,.2,1);
  position:relative;
}
.conf-fill::after {
  content:''; position:absolute; inset:0;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.3),transparent);
  animation:shimmer 2s infinite;
}
@keyframes shimmer { 0%{transform:translateX(-100%)} 100%{transform:translateX(100%)} }
.conf-legend { display:flex; gap:1rem; flex-wrap:wrap; margin-top:.75rem; }
.leg { display:flex; align-items:center; gap:5px; font-size:.68rem; color:var(--text-3); }
.leg-dot { width:8px; height:8px; border-radius:2px; }

/* ── prob bars ── */
.prob-lbl { font-size:.72rem; color:var(--text-2); width:58px; flex-shrink:0; font-family:var(--mono); }
.prob-lbl.top { color:var(--text); font-weight:700; }
.prob-track { flex:1; height:8px; background:var(--border); border-radius:5px; overflow:hidden; }
.prob-fill  { height:100%; border-radius:5px; }
.prob-pct   { font-family:var(--mono); font-size:.7rem; color:var(--text-3); width:42px; text-align:right; flex-shrink:0; }
.prob-pct.top { color:var(--text); font-weight:700; }

/* ── recommendation steps ── */
.rec-step {
  display:flex; align-items:flex-start; gap:12px;
  background:var(--surface-2); border:1px solid var(--border); border-left:3px solid;
  border-radius:var(--radius-sm); padding:.85rem 1rem;
  position:relative; transition:var(--ease); margin-bottom:8px;
}
.rec-step:hover { background:var(--surface); box-shadow:var(--shadow); }
.rec-step.urgent { border-left-color:var(--danger); }
.rec-step.warn   { border-left-color:var(--accent); }
.rec-step.ok     { border-left-color:var(--primary); }
.rec-step-icon  { font-size:1.1rem; flex-shrink:0; margin-top:1px; }
.rec-step-title { font-size:.84rem; font-weight:600; color:var(--text); margin-bottom:3px; }
.rec-step-detail{ font-size:.76rem; color:var(--text-2); line-height:1.6; }
.rec-step-num   { position:absolute; top:8px; right:10px; font-family:var(--mono); font-size:.6rem; color:var(--text-3); opacity:.4; }

/* ── OOD banner ── */
.ood-banner {
  background:var(--warn-bg); border:1px solid #FFD54F; border-left:4px solid var(--warn);
  border-radius:var(--radius-sm); padding:1.25rem 1.5rem;
  display:flex; align-items:flex-start; gap:16px;
}
.ood-icon  { font-size:1.75rem; flex-shrink:0; }
.ood-title { font-size:.95rem; font-weight:700; color:var(--warn); margin-bottom:4px; }
.ood-body  { font-size:.8rem; color:var(--text-2); line-height:1.65; }

/* ── meta pills ── */
.meta-strip { display:flex; gap:.75rem; flex-wrap:wrap; margin:.75rem 0; }
.meta-pill  {
  display:flex; align-items:center; gap:6px;
  background:var(--surface-2); border:1px solid var(--border); border-radius:20px;
  padding:4px 12px; font-size:.72rem; color:var(--text-2); font-family:var(--mono);
}

/* ── batch card ── */
.batch-card {
  background:var(--surface); border:1px solid var(--border);
  border-radius:var(--radius-sm); padding:.7rem; text-align:center; box-shadow:var(--shadow);
}
.mini-bar { width:100%; height:4px; background:var(--border); border-radius:3px; overflow:hidden; margin-top:6px; }
.mini-fill { height:100%; border-radius:3px; }

/* ── history ── */
.hist-item {
  background:var(--surface-2); border:1px solid var(--border); border-radius:var(--radius-sm);
  padding:.65rem .85rem; margin-bottom:6px; transition:var(--ease);
}
.hist-item:hover { background:var(--primary-bg); border-color:var(--primary-bd); }
.hist-pred { font-weight:600; color:var(--text); font-size:.82rem; margin-bottom:2px; }
.hist-meta { color:var(--text-3); font-family:var(--mono); font-size:.65rem; line-height:1.6; }

/* ── model card (sidebar) ── */
.model-info-card {
  background:var(--primary-bg); border:1px solid var(--primary-bd); border-left:3px solid var(--primary);
  border-radius:var(--radius-sm); padding:.85rem 1rem; margin-bottom:1rem;
}
.model-tag  { font-family:var(--mono); font-size:.6rem; font-weight:700; text-transform:uppercase; letter-spacing:.12em; color:var(--primary); margin-bottom:4px; }
.model-name { font-size:.88rem; font-weight:700; color:var(--text); margin-bottom:4px; }
.model-desc { font-size:.8rem; color:var(--text-2); line-height:1.55; }

/* ── tips ── */
.tip {
  background:var(--surface); border:1px solid var(--border); border-radius:var(--radius-sm);
  padding:.9rem 1rem; margin-bottom:8px;
  display:flex; align-items:flex-start; gap:12px; transition:var(--ease);
}
.tip:hover { box-shadow:var(--shadow); }
.tip-icon  { font-size:1.2rem; flex-shrink:0; margin-top:1px; }
.tip-title { font-size:.82rem; font-weight:600; color:var(--text); margin-bottom:2px; }
.tip-body  { font-size:.74rem; color:var(--text-2); line-height:1.55; }

/* ── info/success banners ── */
.info-banner {
  background:var(--info-bg); border:1px solid #90CAF9; border-left:3px solid var(--info);
  border-radius:var(--radius-sm); padding:.75rem 1rem; font-size:.8rem; color:var(--info); margin:.5rem 0;
}
.warn-banner {
  background:var(--accent-bg); border:1px solid #FFE082; border-left:3px solid var(--accent);
  border-radius:var(--radius-sm); padding:.75rem 1rem; font-size:.8rem; color:var(--text-2); margin:.5rem 0;
}

/* ── buttons ── */
.stButton > button {
  background:var(--surface) !important; border:1px solid var(--border) !important;
  color:var(--text-2) !important; border-radius:var(--radius-xs) !important;
  font-family:var(--sans) !important; font-size:.84rem !important; font-weight:500 !important;
  padding:.5rem 1rem !important; transition:var(--ease) !important; box-shadow:var(--shadow) !important;
}
.stButton > button:hover {
  background:var(--primary-bg) !important; border-color:var(--primary) !important;
  color:var(--primary) !important; transform:translateY(-1px) !important; box-shadow:var(--shadow-md) !important;
}
.btn-primary > button {
  background:var(--primary) !important; border-color:var(--primary) !important; color:#fff !important;
}
.btn-primary > button:hover {
  background:var(--primary-lt) !important; border-color:var(--primary-lt) !important; color:#fff !important;
}

/* ── tabs ── */
.stTabs [data-baseweb="tab-list"] { background:transparent !important; border-bottom:2px solid var(--border) !important; gap:0 !important; }
.stTabs [data-baseweb="tab"] {
  background:transparent !important; color:var(--text-3) !important;
  font-family:var(--sans) !important; font-size:.88rem !important; font-weight:500 !important;
  border-bottom:2px solid transparent !important; border-radius:0 !important;
  padding:.6rem 1.25rem !important; margin-bottom:-2px !important; transition:var(--ease) !important;
}
.stTabs [aria-selected="true"] { color:var(--primary) !important; border-bottom-color:var(--primary) !important; font-weight:600 !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top:1.25rem !important; }

/* ── sidebar ── */
div[data-testid="stSidebar"] { background:var(--surface) !important; border-right:1px solid var(--border) !important; }
div[data-testid="stSidebar"] .block-container { padding:1rem !important; }
div[data-testid="stSelectbox"] > div > div { background:var(--surface-2) !important; border-color:var(--border) !important; border-radius:var(--radius-xs) !important; color:var(--text) !important; }
div[data-testid="stSelectbox"] label { color:var(--text) !important; font-size:.84rem !important; font-weight:600 !important; margin-bottom:.3rem !important; display:block !important; }
div[data-testid="stSelectbox"] [data-baseweb="select"] span,
div[data-testid="stSelectbox"] [data-baseweb="select"] div { color:var(--text) !important; }
div[data-testid="stSelectbox"] [data-baseweb="select"] input { color:var(--text) !important; }
div[data-testid="stRadio"] label { font-size:.88rem !important; color:var(--text) !important; }
div[data-testid="stFileUploaderDropzone"] { border:none !important; background:transparent !important; padding:.5rem 1.5rem 1.25rem !important; }
div[data-testid="stFileUploader"] > label { display:none !important; }
div[data-testid="stFileUploader"] section  { background:transparent !important; border:none !important; padding:0 !important; }
.stImage img { border-radius:var(--radius-sm) !important; }
div[data-testid="stProgress"] > div > div { background:linear-gradient(90deg,var(--primary),var(--primary-lt)) !important; border-radius:8px !important; }
div[data-testid="stExpander"] { border:1px solid var(--border) !important; border-radius:var(--radius-sm) !important; background:var(--surface) !important; }

/* ── status dot ── */
.sdot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; animation:sdot-p 2s infinite; }
@keyframes sdot-p { 0%,100%{opacity:1} 50%{opacity:.4} }
.sdot-ok   { background:var(--primary); }
.sdot-warn { background:var(--accent); }

/* ── divider ── */
.cv-hr { border:none; border-top:1px solid var(--border); margin:1.25rem 0; }

/* ── footer ── */
.cv-footer { border-top:1px solid var(--border); padding:2rem 0 0; margin-top:3rem; text-align:center; color:var(--text-3); font-size:.78rem; }

/* ── responsive ── */
@media (max-width:768px) {
  .hero { padding:1.5rem; }
  .hero-title { font-size:1.5rem; }
  .stat-strip { grid-template-columns:repeat(2,1fr); }
  .hero-stats { gap:1rem; }
}

/* ── cassava validation banner ── */
.valid-banner-ok {
  background:#E8F5E9; border:1px solid #A5D6A7; border-left:4px solid var(--primary);
  border-radius:var(--radius-sm); padding:1.1rem 1.5rem;
  display:flex; align-items:flex-start; gap:16px; margin-bottom:1rem;
}
.valid-icon  { font-size:1.45rem; flex-shrink:0; color:var(--primary); font-weight:800; line-height:1.4; }
.valid-title { font-size:.95rem; font-weight:700; color:var(--primary); margin-bottom:4px; }
.valid-body  { font-size:.8rem; color:var(--text-2); line-height:1.65; }
.validity-score-badge {
  display:inline-flex; align-items:center; gap:7px;
  background:var(--surface); border:1px solid var(--border); border-radius:20px;
  padding:4px 14px; font-family:var(--mono); font-size:.78rem; font-weight:600;
  margin:.25rem 0;
}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# UTILITY HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def H(html: str) -> None:
    """Shorthand for st.markdown with unsafe_allow_html=True."""
    st.markdown(html, unsafe_allow_html=True)


def conf_color(p: float) -> str:
    if p >= 0.75: return "#2E7D32"
    if p >= 0.45: return "#FF8F00"
    return "#C62828"


def conf_label(p: float) -> str:
    if p >= 0.75: return "High confidence"
    if p >= 0.45: return "Medium confidence"
    return "Low — re-verify"


def _now() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def fmt_bytes(n: int) -> str:
    if n < 1024:     return f"{n} B"
    if n < 1048576:  return f"{n/1024:.1f} KB"
    return f"{n/1048576:.1f} MB"


def _sev_card_cls(sev: str) -> str:
    return {"High": "high", "Medium": "med", "None": "none"}.get(sev, "none")


def _step_urgency(pred_idx: int, step_i: int) -> str:
    if pred_idx == 4: return "ok"
    if pred_idx == 2: return "warn" if step_i == 0 else "ok"
    return "urgent" if step_i < 2 else "warn"


# ═════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═════════════════════════════════════════════════════════════════════════════

def init_session_state() -> None:
    defaults = {
        "prediction_history":  [],
        "last_prediction":     None,
        "last_probs":          None,
        "last_image":          None,
        "last_validation":     None,
        "batch_results":       None,
        "images_processed":    0,
        "batch_uploader_gen":  0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ═════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═════════════════════════════════════════════════════════════════════════════

def _ckpt_path(display_name: str) -> Optional[Path]:
    meta = MODELS_META[display_name]
    base = Path(MODELS_DIR)
    p    = base / meta["ckpt"]
    return p if p.exists() else None


@st.cache_resource(show_spinner="Loading model weights…")
def load_model(model_key: str, ckpt_str: str):
    """Load a PyTorch checkpoint. Returns (model, device) or (None, 'cpu')."""
    if not _CODES_OK or not _TORCH_OK:
        return None, "cpu"
    try:
        device = get_device()
        model  = build_model(model_key, num_classes=NUM_CLASSES, dropout=0.3)
        state  = torch.load(ckpt_str, map_location=device, weights_only=True)
        if isinstance(state, dict) and "net" in state:
            state = state["net"]
        model.load_state_dict(state, strict=False)
        model.eval().to(device)
        return model, device
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None, "cpu"


# ═════════════════════════════════════════════════════════════════════════════
# CASSAVA IMAGE VALIDATION  (Isolation Forest gatekeeper)
# ═════════════════════════════════════════════════════════════════════════════

# ResNet-50 transform — must match codes/outlier_handler.LeafDataset.TFM
# so that inference features are comparable to the training distribution.
_RESNET_TFM = (
    _tv_transforms.Compose([
        _tv_transforms.Resize(256),
        _tv_transforms.CenterCrop(224),
        _tv_transforms.ToTensor(),
        _tv_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    if _TORCH_OK and _TORCHVISION_OK else None
)


@st.cache_resource(show_spinner="Loading Isolation Forest validator…")
def load_isolation_forest():
    """Load the saved validation bundle from models/isolation_forest.pkl.
    Returns the bundle dict (keys: scaler, pca, iso) or None if unavailable.
    Generate the file first by running: python codes/outlier_handler.py
    """
    if not _JOBLIB_OK:
        return None
    pkl_path = Path(MODELS_DIR) / "isolation_forest.pkl"
    if not pkl_path.exists():
        return None
    try:
        bundle = joblib.load(str(pkl_path))
        # Verify the required key is present
        if "iso" not in bundle:
            return None
        return bundle
    except Exception:
        return None


@st.cache_resource(show_spinner="Loading ResNet-50 feature extractor…")
def load_resnet_extractor():
    """Load ResNet-50 (ImageNet weights) with FC removed → 2048-dim embeddings.
    Returns (backbone, device) or (None, 'cpu') on failure.
    Reuses the same backbone architecture as codes/outlier_handler.py.
    """
    if not _TORCH_OK or not _TORCHVISION_OK:
        return None, "cpu"
    try:
        _dev = get_device() if _CODES_OK else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        backbone = _tv_models.resnet50(weights="IMAGENET1K_V2")
        backbone.fc = _nn.Identity()   # strip classification head → (N, 2048)
        backbone.eval().to(_dev)
        return backbone, _dev
    except Exception:
        return None, "cpu"


def extract_resnet_features(image: Image.Image) -> Optional[np.ndarray]:
    """Extract a 2048-dim ResNet-50 feature vector from a PIL image.
    Returns np.ndarray of shape (2048,) or None on failure.
    """
    if _RESNET_TFM is None:
        return None
    backbone, dev = load_resnet_extractor()
    if backbone is None:
        return None
    try:
        tensor = _RESNET_TFM(image.convert("RGB")).unsqueeze(0).to(dev)
        with torch.no_grad():
            feat = backbone(tensor).squeeze().cpu().numpy()
        return feat   # shape (2048,)
    except Exception:
        return None


def validate_cassava_image(image: Image.Image) -> dict:
    """Validate whether an uploaded image is a cassava leaf.

    Pipeline:
        PIL Image
          → ResNet-50 (2048-dim)
          → StandardScaler (from training bundle)
          → PCA (64-dim, from training bundle)
          → IsolationForest.decision_function()
          → threshold check (OOD_THRESHOLD)

    Returns:
        {
            "is_cassava":          bool   — True = pass through to disease models
            "score":               float  — raw IF anomaly score
            "validity_percentage": float  — 0–100 user-friendly display score
            "method":              str    — "isolation_forest" | "unavailable"
        }

    When the pkl is missing (first run before training), method="unavailable"
    and is_cassava=True so the app degrades gracefully without blocking users.
    """
    # Graceful default: pass through when validation model is unavailable
    _default = {
        "is_cassava":          True,
        "score":               0.0,
        "validity_percentage": 100.0,
        "method":              "unavailable",
    }

    bundle = load_isolation_forest()
    if bundle is None:
        return _default

    feat = extract_resnet_features(image)
    if feat is None:
        return _default

    try:
        iso = bundle["iso"]

        # Apply the same scaler + PCA that were fitted during training
        x = feat.reshape(1, -1)   # (1, 2048)
        if bundle.get("scaler") is not None:
            x = bundle["scaler"].transform(x)
        if bundle.get("pca") is not None:
            x = bundle["pca"].transform(x)    # (1, 64)

        score = float(iso.decision_function(x)[0])

        # Convert raw score to a 0–100 "cassava validity" percentage.
        # Scores for in-distribution (cassava) images cluster near 0 to +0.2.
        # Scores for out-of-distribution images are typically below -0.2.
        # Formula maps [-0.2, +0.2] → [0%, 100%] and clips outside that range.
        validity_pct = max(0.0, min(100.0, ((score + 0.2) / 0.4) * 100))

        return {
            "is_cassava":          score >= OOD_THRESHOLD,
            "score":               score,
            "validity_percentage": validity_pct,
            "method":              "isolation_forest",
        }
    except Exception:
        return _default


# ═════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═════════════════════════════════════════════════════════════════════════════

def predict_image(
    image: Image.Image,
    model,
    device,
) -> tuple[int, np.ndarray]:
    """
    Run inference on a PIL image.
    Returns (predicted_class_index, probability_array).
    Falls back to seeded-random dummy when no real model is available.
    """
    if not _CODES_OK or model is None:
        # Deterministic dummy: same image → same prediction
        seed = int(np.array(image.resize((32, 32))).mean() * 1000) % (2**31)
        rng  = np.random.default_rng(seed)
        probs = rng.dirichlet(np.ones(NUM_CLASSES) * 2.0)
        top   = int(np.argmax(probs))
        probs[top] += 0.30          # boost top class so it looks realistic
        probs /= probs.sum()
        return top, probs
    if not _TORCH_OK:
        rng   = np.random.default_rng(42)
        probs = rng.dirichlet(np.ones(NUM_CLASSES))
        return int(np.argmax(probs)), probs
    tfm    = get_transforms(INPUT_SIZE, augment=False)
    tensor = tfm(image=np.array(image.convert("RGB")))["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    return int(np.argmax(probs)), probs


def is_non_cassava(probs: np.ndarray, model_is_real: bool) -> bool:
    """
    Detect out-of-distribution images (non-cassava leaves).

    Strategy (only when real model is loaded):
      • If max class probability is below OOD_MAX_PROB  → model is very uncertain
      • If normalised entropy is above OOD_ENTROPY       → distribution is near-uniform
    Either condition triggers the OOD flag.

    In demo mode (dummy predictions) we do NOT flag anything as OOD because
    random predictions cannot reliably represent true model uncertainty.
    """
    if not model_is_real:
        return False
    max_p = float(np.max(probs))
    n     = len(probs)
    ent   = -float(np.sum(probs * np.log(probs + 1e-12))) / np.log(n)
    return max_p < OOD_MAX_PROB or ent > OOD_ENTROPY


# ═════════════════════════════════════════════════════════════════════════════
# GRAD-CAM
# ═════════════════════════════════════════════════════════════════════════════

def generate_gradcam(
    image: Image.Image,
    model,
    device,
    pred_idx: int,
    model_key: str,
    alpha: float = 0.5,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Return (overlay_rgb, original_resized_rgb).
    Uses real Grad-CAM when codes/ is available; falls back to a simulated
    Gaussian-blob heatmap that can be swapped for the real implementation later.
    """
    img_rgb = np.array(image.convert("RGB"))
    # ── real Grad-CAM ─────────────────────────────────────────────────────────
    if _CODES_OK and model is not None:
        try:
            inp, resized = preprocess_for_gradcam(img_rgb, INPUT_SIZE)
            cam          = GradCAM(model, model_name=model_key)
            heatmap, _   = cam(inp.to(device), class_idx=pred_idx)
            cam.remove_hooks()
            return overlay_heatmap(resized, heatmap, alpha=alpha), resized
        except Exception:
            pass  # fall through to simulation
    # ── simulated Grad-CAM ───────────────────────────────────────────────────
    if not (_CV2_OK and _MPL_OK):
        return None, None
    try:
        s     = INPUT_SIZE
        small = cv2.resize(img_rgb, (s, s)).astype(np.float32) / 255.0
        rng   = np.random.default_rng(seed=pred_idx * 7 + 13)
        heat  = np.zeros((s, s), dtype=np.float32)
        n_blobs = 3 if pred_idx != 4 else 1
        for _ in range(n_blobs):
            cx = int(rng.integers(s // 4, 3 * s // 4))
            cy = int(rng.integers(s // 4, 3 * s // 4))
            r  = int(rng.integers(s // 6, s // 3))
            ys, xs = np.ogrid[:s, :s]
            heat  += np.exp(-((xs-cx)**2 + (ys-cy)**2) / (2*r**2)) * float(rng.uniform(.6, 1.))
        heat    = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        h_color = cv2.applyColorMap((heat*255).astype(np.uint8), cv2.COLORMAP_JET)
        h_rgb   = cv2.cvtColor(h_color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        overlay = np.clip((1-alpha)*small + alpha*h_rgb, 0, 1)
        return (overlay*255).astype(np.uint8), (small*255).astype(np.uint8)
    except Exception:
        return None, None


# ═════════════════════════════════════════════════════════════════════════════
# GOOGLE DRIVE DOWNLOAD
# ═════════════════════════════════════════════════════════════════════════════

def _gdrive_id(url: str) -> Optional[str]:
    for pat in [r"/file/d/([a-zA-Z0-9_-]+)", r"[?&]id=([a-zA-Z0-9_-]+)"]:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return None


def download_from_google_drive(url: str) -> tuple[list[Image.Image], str]:
    """Download a single image or ZIP from a Google Drive share link."""
    fid = _gdrive_id(url.strip())
    if not fid:
        return [], "Could not extract a file ID. Paste a full Google Drive share link."
    if not _REQUESTS_OK:
        return [], "`requests` is not installed. Run: pip install requests"
    try:
        sess     = _req.Session()
        base_url = f"https://drive.google.com/uc?export=download&id={fid}"
        resp     = sess.get(base_url, stream=True, timeout=30)
        for k, v in resp.cookies.items():
            if k.startswith("download_warning"):
                resp = sess.get(base_url + f"&confirm={v}", stream=True, timeout=60)
                break
        if resp.status_code != 200 or len(resp.content) < 512:
            return [], "Download failed. Make sure the file is shared with 'Anyone with the link'."
        data = resp.content
    except Exception as exc:
        return [], f"Network error: {exc}"

    if data[:2] == b"PK":
        imgs = _zip_images(data)
        return (imgs, "") if imgs else ([], "ZIP contained no valid images.")
    try:
        return [Image.open(io.BytesIO(data)).convert("RGB")], ""
    except Exception:
        return [], "Could not open the downloaded file as an image."


def _zip_images(raw: bytes) -> list[Image.Image]:
    imgs: list[Image.Image] = []
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            for name in zf.namelist():
                if Path(name).suffix.lower() in VALID_EXTS:
                    try:
                        imgs.append(Image.open(io.BytesIO(zf.read(name))).convert("RGB"))
                    except Exception:
                        pass
    except zipfile.BadZipFile:
        pass
    return imgs


# ═════════════════════════════════════════════════════════════════════════════
# BATCH PROCESSING
# ═════════════════════════════════════════════════════════════════════════════

def process_batch(
    uploaded_files: list,
    display_name: str,
    model,
    device,
    crop: str,
) -> Optional[object]:
    """
    Process a list of uploaded files (images and/or ZIPs).
    Returns a pandas DataFrame with one row per image.
    """
    if not _PANDAS_OK:
        st.error("pandas is required for batch processing.")
        return None

    # Collect (image, filename) pairs from all uploaded files
    pairs: list[tuple[Image.Image, str]] = []
    for uf in uploaded_files:
        mb = uf.size / (1024 * 1024)
        if mb > MAX_ZIP_MB:
            st.warning(f"⚠ {uf.name} skipped — too large ({mb:.1f} MB).")
            continue
        raw    = uf.read()
        is_zip = uf.name.lower().endswith(".zip")
        if is_zip:
            try:
                with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                    for name in zf.namelist():
                        if Path(name).suffix.lower() in VALID_EXTS:
                            try:
                                img = Image.open(io.BytesIO(zf.read(name))).convert("RGB")
                                pairs.append((img, Path(name).name))
                            except Exception:
                                pass
            except zipfile.BadZipFile:
                st.warning(f"⚠ {uf.name} is not a valid ZIP — skipped.")
        else:
            try:
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                pairs.append((img, uf.name))
            except Exception:
                st.warning(f"⚠ {uf.name} could not be read as an image — skipped.")

    if not pairs:
        st.warning("No valid images found in the uploaded files.")
        return None

    rows: list[dict] = []
    total = len(pairs)
    prog  = st.progress(0, text="Starting batch…")

    for i, (img, fname) in enumerate(pairs):
        pred_idx, probs = predict_image(img, model, device)
        ood      = is_non_cassava(probs, model_is_real=(model is not None))
        conf_val = float(probs[pred_idx])
        rows.append({
            "Filename":   fname,
            "Crop":       crop,
            "Model":      display_name,
            "Prediction": "Non-cassava (flagged)" if ood else CLASS_NAMES[pred_idx],
            "Confidence": round(conf_val * 100, 1) if not ood else 0.0,
            "Severity":   "—" if ood else SEVERITY[pred_idx],
            "Timestamp":  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        prog.progress((i + 1) / total, text=f"Processed {i+1}/{total}: {fname}")
        time.sleep(0.04)

    prog.empty()
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# PLOTLY CHARTS
# ═════════════════════════════════════════════════════════════════════════════

def make_gauge(confidence: float) -> Optional[object]:
    """Circular confidence gauge using Plotly Indicator."""
    if not _PLOTLY_OK:
        return None
    pct = confidence * 100
    col = conf_color(confidence)
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = pct,
        delta = {"reference": 75, "suffix": "%", "valueformat": ".1f"},
        number= {"suffix": "%", "font": {"size": 38, "color": col, "family": "JetBrains Mono"}},
        gauge = {
            "axis":  {"range": [0, 100], "tickwidth": 1, "tickcolor": "#90A4AE",
                      "tickfont": {"family": "JetBrains Mono", "size": 10}},
            "bar":   {"color": col, "thickness": 0.28},
            "bgcolor": "#F5F7FA", "bordercolor": "#E0E0E0",
            "steps": [
                {"range": [0,  45], "color": "#FFEBEE"},
                {"range": [45, 75], "color": "#FFF8E1"},
                {"range": [75,100], "color": "#E8F5E9"},
            ],
            "threshold": {"line": {"color": "#263238", "width": 2}, "thickness": 0.75, "value": pct},
        },
        title={"text": "Confidence Score", "font": {"size": 13, "color": "#546E7A", "family": "Inter"}},
    ))
    fig.update_layout(height=240, margin=dict(l=20,r=20,t=30,b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def make_prob_chart(probs: np.ndarray) -> Optional[object]:
    """Horizontal bar chart of class probabilities."""
    if not _PLOTLY_OK:
        return None
    labels = [CLASS_SHORT[i] for i in range(len(probs))]
    values = [float(p*100) for p in probs]
    colors = [conf_color(p/100) for p in values]
    fig = go.Figure(go.Bar(
        y=labels, x=values, orientation="h",
        marker_color=colors, marker_line_width=0,
        text=[f"{v:.1f}%" for v in values], textposition="outside",
        textfont={"family": "JetBrains Mono", "size": 11, "color": "#546E7A"},
        hovertemplate="<b>%{y}</b><br>%{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        height=220, margin=dict(l=10,r=50,t=30,b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Probability (%)", range=[0,115], showgrid=True, gridcolor="#F0F0F0",
                   tickfont={"family":"JetBrains Mono","size":10}),
        yaxis=dict(autorange="reversed", tickfont={"family":"JetBrains Mono","size":11,"color":"#263238"}),
        title=dict(text="Class Probabilities", font={"size":13,"color":"#546E7A","family":"Inter"}),
        bargap=0.35,
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# UI — HEADER
# ═════════════════════════════════════════════════════════════════════════════

def render_header() -> None:
    n = st.session_state.get("images_processed", 0)
    H(f"""
    <div class="hero">
      <div class="hero-inner">
        <div class="hero-brand">
          <div class="hero-logo">🌿</div>
          <div>
            <div class="hero-title">Cassava Vision AI</div>
            <div class="hero-sub">AI-Powered Crop Disease Detection &amp; Explainable Diagnostics Platform</div>
          </div>
        </div>
        <div class="hero-stats">
          <div><div class="hero-stat-v">{n}</div><div class="hero-stat-l">Images Processed</div></div>
          <div><div class="hero-stat-v">3</div><div class="hero-stat-l">Models Available</div></div>
          <div><div class="hero-stat-v">{len(CROPS)}</div><div class="hero-stat-l">Crops Supported</div></div>
          <div><div class="hero-stat-v">5</div><div class="hero-stat-l">Disease Classes</div></div>
        </div>
      </div>
    </div>
    """)


def render_stat_strip(model_ready: bool) -> None:
    dot   = "sdot-ok" if model_ready else "sdot-warn"
    label = "Model online" if model_ready else "Demo mode"
    H(f"""
    <div class="stat-strip">
      <div class="stat-card"><div class="stat-icon g">🦠</div>
        <div><div class="stat-val">5</div><div class="stat-lbl">Disease Classes</div></div></div>
      <div class="stat-card"><div class="stat-icon o">📷</div>
        <div><div class="stat-val">21K+</div><div class="stat-lbl">Training Images</div></div></div>
      <div class="stat-card"><div class="stat-icon b">🧠</div>
        <div><div class="stat-val">3</div><div class="stat-lbl">AI Models</div></div></div>
      <div class="stat-card"><div class="stat-icon r">
        <span class="sdot {dot}" style="margin:0"></span></div>
        <div><div class="stat-val" style="font-size:1rem">{label}</div>
        <div class="stat-lbl">System Status</div></div></div>
    </div>
    """)


# ═════════════════════════════════════════════════════════════════════════════
# UI — SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> tuple[str, str, bool, object, object]:
    """
    Render the sidebar.
    Returns (display_name, crop, model_ready, model, device).
    """
    with st.sidebar:
        # ── brand ──────────────────────────────────────────────────────────
        H("""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:1.25rem;
          padding-bottom:1rem;border-bottom:1px solid var(--border)">
          <span style="font-size:1.75rem">🌿</span>
          <div>
            <div style="font-size:.95rem;font-weight:700;color:var(--text)">Cassava Vision AI</div>
            <div style="font-size:.68rem;color:var(--text-3);font-family:var(--mono)">Agricultural AI</div>
          </div>
        </div>
        """)
        # ── model selection ─────────────────────────────────────────────────
        H('<div class="sec-label"><span class="sec-num">1</span>Select Model</div>')
        display_names = list(MODELS_META.keys())
        selected_display = st.selectbox(
            "Select AI Model", display_names,
            index=0, key="model_select", label_visibility="visible",
        )
        meta = MODELS_META[selected_display]
        H(f"""
        <div class="model-info-card">
          <div class="model-tag">{meta['tag']} · {meta['acc']} accuracy · {meta['params']}</div>
          <div class="model-name">{selected_display}</div>
          <div class="model-desc">{meta['desc']}</div>
        </div>
        """)
        # ── crop selection ──────────────────────────────────────────────────
        H('<div class="sec-label"><span class="sec-num">2</span>Select Crop</div>')
        crop = st.selectbox(
            "Select Crop Type", CROPS, index=0, key="crop_select", label_visibility="visible",
        )
        if crop != "Cassava":
            H(f"""
            <div class="info-banner">
              ℹ️ Model is trained on Cassava. Results for <strong>{crop}</strong>
              are indicative only — crop-specific models coming soon.
            </div>
            """)

        H('<hr class="cv-hr">')
        # ── load model / status ─────────────────────────────────────────────
        ckpt        = _ckpt_path(selected_display)
        model_ready = ckpt is not None
        model, device = None, "cpu"
        if model_ready:
            model, device = load_model(meta["key"], str(ckpt))
            if model is None:
                model_ready = False

        dot   = "sdot-ok"   if model_ready else "sdot-warn"
        label = "Checkpoint loaded" if model_ready else "Demo mode (no checkpoint)"
        H(f"""
        <div style="margin-bottom:1rem">
          <div class="sec-label" style="margin-bottom:.5rem">System Status</div>
          <div style="font-size:.78rem;color:var(--text-2)">
            <span class="sdot {dot}"></span>{label}
          </div>
          {'<div style="font-size:.68rem;color:var(--text-3);font-family:var(--mono);margin-top:4px">' + ckpt.name + '</div>' if ckpt else ''}
        </div>
        """)
        if not model_ready:
            H("""
            <div class="warn-banner">
              <strong style="color:var(--accent)">Demo Mode</strong><br>
              Place <code>.pth</code> checkpoints in <code>models/</code>
              to enable real inference. Predictions are currently simulated.
            </div>
            """)

        H('<hr class="cv-hr">')
        # ── prediction history ──────────────────────────────────────────────
        H('<div class="sec-label"><span class="sec-num">3</span>Recent Predictions</div>')
        history = [e for e in st.session_state.get("prediction_history", []) if e["crop"] == crop]
        if history:
            for entry in reversed(history[-MAX_HISTORY:]):
                col = conf_color(entry["confidence"] / 100)
                H(f"""
                <div class="hist-item">
                  <div class="hist-pred" style="color:{col}">{entry['prediction']}</div>
                  <div class="hist-meta">
                    {entry['filename']} · {entry['crop']}<br>
                    {entry['model']} · {entry['confidence']:.1f}% · {entry['time']}
                  </div>
                </div>
                """)
            if st.button("Clear History", key="btn_clear_hist"):
                st.session_state["prediction_history"] = []
                st.rerun()
        else:
            H('<div style="font-size:.78rem;color:var(--text-3)">No predictions yet.</div>')

        H('<hr class="cv-hr">')
        # ── disease class reference ─────────────────────────────────────────
        with st.expander("Disease Classes Reference"):
            for i in range(NUM_CLASSES):
                sc  = SEVERITY_COLOR[SEVERITY[i]]
                url = DISEASE_LINKS[i]
                H(f"""
                <div style="display:flex;align-items:flex-start;gap:8px;margin-bottom:10px;padding-bottom:10px;border-bottom:1px solid #E0E0E0;">
                  <span style="width:8px;height:8px;border-radius:50%;background:{sc};
                    display:inline-block;flex-shrink:0;margin-top:4px;"></span>
                  <div>
                    <div style="font-size:.75rem;margin-bottom:2px;">
                      <strong style="color:#263238">{CLASS_SHORT[i]}</strong>
                      <span style="color:#546E7A"> — {CLASS_NAMES[i]}</span>
                    </div>
                    <div style="font-size:.7rem;color:#90A4AE;margin-bottom:3px;">{CLASS_DESCRIPTIONS[i]}</div>
                    <a href="{url}" target="_blank" style="font-size:.68rem;color:#1565C0;text-decoration:none;font-family:monospace;">
                      📖 Learn more &rarr;
                    </a>
                  </div>
                </div>
                """)

        H("""
        <div style="font-size:.68rem;color:var(--text-3);line-height:1.65;
          margin-top:.5rem;padding-top:.75rem;border-top:1px solid var(--border)">
          For research &amp; educational use only.<br>
          Always verify with a qualified agronomist.
        </div>
        """)

    return selected_display, crop, model_ready, model, device


# ═════════════════════════════════════════════════════════════════════════════
# UI — RESULTS COMPONENTS
# ═════════════════════════════════════════════════════════════════════════════

def render_image_meta(image: Image.Image, filename: str, file_bytes: int) -> None:
    w, h = image.size
    H(f"""
    <div class="meta-strip">
      <div class="meta-pill">📁 {filename}</div>
      <div class="meta-pill">📐 {w}×{h} px</div>
      <div class="meta-pill">💾 {fmt_bytes(file_bytes)}</div>
      <div class="meta-pill">🎨 {image.mode}</div>
    </div>
    """)


def render_prediction_card(pred_idx: int, probs: np.ndarray, crop: str) -> None:
    top_p   = float(probs[pred_idx])
    col     = conf_color(top_p)
    sev     = SEVERITY[pred_idx]
    sev_col = SEVERITY_COLOR[sev]
    css_cls = _sev_card_cls(sev)
    H(f"""
    <div class="pred-card {css_cls}">
      <div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:.75rem">
        <div>
          <div class="pred-eyebrow">Predicted Condition · {crop}</div>
          <div class="pred-name">{CLASS_NAMES[pred_idx]}</div>
          <div class="sev-badge" style="background:{sev_col}18;border:1px solid {sev_col}40">
            <span style="width:7px;height:7px;border-radius:50%;background:{sev_col};display:inline-block"></span>
            <span style="color:{sev_col}">Severity: {sev}</span>
          </div>
        </div>
        <div class="conf-pill" style="background:{col}15;color:{col};border:1px solid {col}40">
          <span class="conf-dot" style="background:{col}"></span>
          {top_p*100:.1f}%
        </div>
      </div>
      <div class="pred-desc">{CLASS_DESCRIPTIONS[pred_idx]}</div>
    </div>
    """)


def render_confidence_section(probs: np.ndarray, pred_idx: int) -> None:
    top_p = float(probs[pred_idx])
    col   = conf_color(top_p)
    pct   = top_p * 100
    # Animated HTML progress bar
    H(f"""
    <div class="conf-wrap">
      <div class="conf-header">
        <span class="conf-lbl">Confidence Score</span>
        <span class="conf-val" style="color:{col}">{pct:.1f}%</span>
      </div>
      <div class="conf-bar">
        <div class="conf-fill" style="width:{pct:.1f}%;background:linear-gradient(90deg,{col},{col}bb)"></div>
      </div>
      <div class="conf-legend">
        <span class="leg"><span class="leg-dot" style="background:#C62828"></span>0–44% Low</span>
        <span class="leg"><span class="leg-dot" style="background:#FF8F00"></span>45–74% Medium</span>
        <span class="leg"><span class="leg-dot" style="background:#2E7D32"></span>75–100% High</span>
      </div>
    </div>
    """)
    # Per-class probability bars
    H('<div style="margin-top:.25rem"><div style="font-size:.62rem;font-weight:700;text-transform:uppercase;'
      'letter-spacing:.12em;color:var(--text-3);margin-bottom:.85rem;font-family:var(--mono)">'
      'Class Probability Distribution</div>')
    for i in range(NUM_CLASSES):
        p  = float(probs[i]) * 100
        bc = conf_color(probs[i])
        op = "" if i == pred_idx else "88"
        is_top = i == pred_idx
        H(f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
          <div class="prob-lbl {'top' if is_top else ''}">{CLASS_SHORT[i]}</div>
          <div class="prob-track">
            <div class="prob-fill" style="width:{max(p,1):.1f}%;background:{bc}{op}"></div>
          </div>
          <div class="prob-pct {'top' if is_top else ''}">{p:.1f}%</div>
        </div>
        """)
    H('</div>')


def render_plotly_charts(probs: np.ndarray, pred_idx: int) -> None:
    if not _PLOTLY_OK:
        H('<div class="info-banner">Install plotly for interactive charts: <code>pip install plotly</code></div>')
        return
    c1, c2 = st.columns(2)
    with c1:
        fig = make_gauge(float(probs[pred_idx]))
        if fig: st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with c2:
        fig = make_prob_chart(probs)
        if fig: st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_recommendations(pred_idx: int) -> None:
    _BORDER = {"urgent": "#C62828", "warn": "#FF8F00", "ok": "#2E7D32"}
    steps_html = ""
    for i, s in enumerate(RECOMMENDATION_STEPS[pred_idx]):
        urg = _step_urgency(pred_idx, i)
        bc  = _BORDER[urg]
        steps_html += (
            f'<div style="display:flex;align-items:flex-start;gap:12px;'
            f'background:#FAFAFA;border:1px solid #E0E0E0;border-left:3px solid {bc};'
            f'border-radius:10px;padding:.85rem 1rem;position:relative;margin-bottom:8px;">'
            f'<div style="font-size:1.1rem;flex-shrink:0;margin-top:1px;">{s["icon"]}</div>'
            f'<div>'
            f'<div style="font-size:.84rem;font-weight:600;color:#263238;margin-bottom:3px;">{s["step"]}</div>'
            f'<div style="font-size:.76rem;color:#546E7A;line-height:1.6;">{s["detail"]}</div>'
            f'</div>'
            f'<div style="position:absolute;top:8px;right:10px;font-family:monospace;'
            f'font-size:.6rem;color:#90A4AE;opacity:.4;">0{i+1}</div>'
            f'</div>'
        )
    H(
        f'<div style="margin-top:1.25rem">'
        f'<div style="font-size:.68rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:.12em;color:#2E7D32;font-family:monospace;margin-bottom:.85rem">'
        f'📋 Recommended Actions</div>'
        f'{steps_html}'
        f'<div style="margin-top:1rem;padding:.9rem 1rem;background:#FAFAFA;'
        f'border:1px solid #E0E0E0;border-radius:10px;font-size:.8rem;color:#546E7A;line-height:1.7">'
        f'<strong style="color:#263238;font-family:monospace;font-size:.68rem;'
        f'letter-spacing:.08em;text-transform:uppercase">Full Treatment Protocol</strong>'
        f'<br><br>{TREATMENTS[pred_idx]}</div>'
        f'</div>'
    )


def render_gradcam_section(
    image: Image.Image, model, device, pred_idx: int, model_key: str
) -> None:
    alpha = st.slider(
        "Heatmap opacity", min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        key="gradcam_alpha",
    )
    with st.spinner("Generating Grad-CAM…"):
        overlay, resized = generate_gradcam(image, model, device, pred_idx, model_key, alpha=alpha)
    if overlay is None:
        H('<div class="info-banner">Grad-CAM requires opencv-python and matplotlib.</div>')
        return
    mode = "Real Grad-CAM" if (_CODES_OK and model is not None) else "Simulated Heatmap"
    c1, c2 = st.columns(2)
    with c1:
        H('<div class="sec-label" style="margin-top:0">Original Image</div>')
        st.image(resized, use_container_width=True)
    with c2:
        H(f'<div class="sec-label" style="margin-top:0">Grad-CAM Overlay ({mode})</div>')
        st.image(overlay, use_container_width=True)
    H(f"""
    <div class="info-banner" style="margin-top:.75rem">
      <strong>Grad-CAM</strong> highlights leaf regions that influenced the prediction.
      <span style="color:#E53935"><strong>Red/Yellow</strong></span> = high influence ·
      <span style="color:#1565C0"><strong>Blue</strong></span> = low influence ·
      Explaining: <strong>{CLASS_NAMES[pred_idx]}</strong>
    </div>
    """)


# ═════════════════════════════════════════════════════════════════════════════
# UI — SHARED ACTION PANEL (preview + Analyze / Retake buttons)
# ═════════════════════════════════════════════════════════════════════════════

def _action_panel(
    image_obj:   Image.Image,
    filename:    str,
    file_bytes:  int,
    analyze_key: str,
    reset_key:   str,
    clear_keys:  list,
    model_ready: bool,
    is_camera:   bool = False,
) -> bool:
    c_prev, c_act = st.columns([2, 3])
    with c_prev:
        caption = "📷 Captured Photo" if is_camera else "Preview"
        st.image(image_obj, use_container_width=True, caption=caption)
    with c_act:
        H('<div style="height:.5rem"></div>')
        render_image_meta(image_obj, filename, file_bytes)
        ca, cr = st.columns(2)
        do_analyze = False
        with ca:
            st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
            label = "🔍 Analyze Photo" if is_camera else "🔍 Analyze Image"
            if st.button(label, key=analyze_key, use_container_width=True):
                do_analyze = True
            st.markdown('</div>', unsafe_allow_html=True)
        with cr:
            rl = "📷 Retake" if is_camera else "↺ Reset"
            if st.button(rl, key=reset_key, use_container_width=True):
                for k in clear_keys + ["last_prediction", "last_probs", "last_image", "last_validation"]:
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()
        if not model_ready:
            H('<div class="info-banner" style="margin-top:.5rem">ℹ️ Demo inference — results are simulated.</div>')
    return do_analyze


# ═════════════════════════════════════════════════════════════════════════════
# UI — SINGLE IMAGE ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

def render_single_analysis(
    display_name: str, crop: str, model, device, model_ready: bool
) -> None:
    H('<div class="sec-label"><span class="sec-num">A</span>Single Image Analysis</div>')

    input_mode = st.radio(
        "Input mode",
        ["📁 Upload File", "📷 Take Photo", "🔗 Google Drive"],
        horizontal=True, key="single_mode", label_visibility="collapsed",
    )

    image_obj: Optional[Image.Image] = None
    filename   = "image"
    file_bytes = 0
    do_analyze = False
    # ── Upload File ──────────────────────────────────────────────────────────
    if "Upload" in input_mode:
        H("""
        <div class="upload-zone">
          <div class="upload-zone-hdr">
            <div class="upload-icon">🌿</div>
            <div class="upload-title">Drop Your Crop Leaf Image Here</div>
            <div class="upload-hint">Drag &amp; drop or click to browse</div>
            <div class="chip-row">
              <span class="chip">JPG</span><span class="chip">JPEG</span>
              <span class="chip">PNG</span><span class="chip">WEBP</span>
            </div>
          </div>
        """)
        uf = st.file_uploader("Upload", type=["jpg","jpeg","png","webp"],
                              key="single_uploader", label_visibility="collapsed")
        H('</div>')
        if uf:
            raw        = uf.read()
            image_obj  = Image.open(io.BytesIO(raw)).convert("RGB")
            filename   = uf.name
            file_bytes = len(raw)
            st.toast("Image uploaded successfully!", icon="✅")
            do_analyze = _action_panel(
                image_obj, filename, file_bytes,
                analyze_key="btn_an_upload", reset_key="btn_rst_upload",
                clear_keys=["single_uploader"], model_ready=model_ready,
            )
    # ── Take Photo ───────────────────────────────────────────────────────────
    elif "Photo" in input_mode:
        H("""
        <div class="camera-zone">
          <div class="camera-hdr">
            <div class="cam-pulse"></div>
            <div class="camera-title">LIVE CAMERA · Point at your crop leaf and capture</div>
          </div>
          <div class="camera-body">
        """)
        snap = st.camera_input("Take photo", key="camera_snap", label_visibility="collapsed")
        H("</div></div>")
        if snap:
            raw        = snap.getvalue()
            image_obj  = Image.open(io.BytesIO(raw)).convert("RGB")
            filename   = "camera_capture.jpg"
            file_bytes = len(raw)
            H('<div style="height:.75rem"></div>')
            do_analyze = _action_panel(
                image_obj, filename, file_bytes,
                analyze_key="btn_an_camera", reset_key="btn_rst_camera",
                clear_keys=["camera_snap"], model_ready=model_ready, is_camera=True,
            )
        else:
            H("""
            <div class="tip" style="margin-top:.75rem">
              <div class="tip-icon">📷</div>
              <div>
                <div class="tip-title">Camera ready — waiting for capture</div>
                <div class="tip-body">Allow camera access if prompted. Point at the crop leaf
                and press the capture button, then click <strong>Analyze Photo</strong>.</div>
              </div>
            </div>
            """)
    # ── Google Drive ─────────────────────────────────────────────────────────
    else:
        H("""
        <div class="upload-zone">
          <div class="upload-zone-hdr">
            <div class="upload-icon">🔗</div>
            <div class="upload-title">Load from Google Drive</div>
            <div class="upload-hint">Paste a shareable link · JPG · PNG · WEBP · ZIP</div>
          </div>
        """)
        H('<div style="padding:0 1.5rem .5rem;font-size:.8rem;color:var(--text-2)">Share with <em>Anyone with the link</em>.</div>')
        url = st.text_input("Drive URL", placeholder="https://drive.google.com/file/d/…/view",
                            key="gdrive_url", label_visibility="collapsed")
        H('</div>')
        if url:
            with st.spinner("Downloading from Google Drive…"):
                imgs, err = download_from_google_drive(url)
            if err:
                st.error(err)
            elif imgs:
                image_obj  = imgs[0]
                filename   = "gdrive_image.jpg"
                file_bytes = 0
                st.toast("Image loaded from Google Drive!", icon="✅")
                do_analyze = _action_panel(
                    image_obj, filename, file_bytes,
                    analyze_key="btn_an_gdrive", reset_key="btn_rst_gdrive",
                    clear_keys=["gdrive_url"], model_ready=model_ready,
                )
    # ── empty state tip ───────────────────────────────────────────────────────
    if image_obj is None and "Photo" not in input_mode:
        if st.session_state.get("last_prediction") is None:
            H("""
            <div class="tip" style="margin-top:.75rem">
              <div class="tip-icon">💡</div>
              <div>
                <div class="tip-title">Provide a crop leaf image to get started</div>
                <div class="tip-body">Upload a file, take a photo, or paste a Google Drive link.
                Choose a clear, well-lit shot where the leaf fills the frame.</div>
              </div>
            </div>
            """)
        if image_obj is None:
            # Still show cached results if they exist
            _show_cached_results(model, device, display_name)
            return
    # ── run inference ─────────────────────────────────────────────────────────
    if do_analyze and image_obj is not None:
        _run_inference(image_obj, filename, crop, display_name, model, device)
    # ── show results ──────────────────────────────────────────────────────────
    _show_cached_results(model, device, display_name)


def _run_inference(
    image_obj: Image.Image,
    filename: str,
    crop: str,
    display_name: str,
    model,
    device,
) -> None:
    """Two-stage inference pipeline:
      1. Cassava Image Validation  — Isolation Forest gates the image.
      2. Disease Classification    — runs only when validation passes.
    """
    st.session_state["last_image"] = image_obj
    # ── Stage 1: Cassava image validation ────────────────────────────────────
    with st.spinner("Validating image…"):
        validation = validate_cassava_image(image_obj)
    st.session_state["last_validation"] = validation

    # When the Isolation Forest is loaded and the image is rejected, stop here.
    # Disease models, Grad-CAM, and recommendations must NOT execute.
    if validation["method"] == "isolation_forest" and not validation["is_cassava"]:
        st.session_state["last_prediction"] = None
        st.session_state["last_probs"]      = None
        st.session_state["images_processed"] = st.session_state.get("images_processed", 0) + 1
        st.toast("Image rejected — does not appear to be a cassava leaf.", icon="⚠️")
        st.rerun()
        return
    # ── Stage 2: Disease classification ──────────────────────────────────────
    with st.spinner("Running disease inference…"):
        st.toast("Running inference…", icon="🔄")
        pred_idx, probs = predict_image(image_obj, model, device)
        ood = is_non_cassava(probs, model_is_real=(model is not None))
    st.session_state["last_prediction"] = (pred_idx, ood)
    st.session_state["last_probs"]      = probs
    st.session_state["images_processed"] = st.session_state.get("images_processed", 0) + 1

    if not ood:
        hist = st.session_state.get("prediction_history", [])
        hist.append({
            "filename":   filename,
            "crop":       crop,
            "model":      display_name,
            "prediction": CLASS_NAMES[pred_idx],
            "confidence": float(probs[pred_idx]) * 100,
            "time":       _now(),
        })
        st.session_state["prediction_history"] = hist[-MAX_HISTORY:]

    st.toast("Analysis completed!", icon="✅")
    st.rerun()


def _show_cached_results(model, device, display_name: str) -> None:
    """Display the last cached inference result, preceded by the validation banner."""
    img        = st.session_state.get("last_image")
    result     = st.session_state.get("last_prediction")
    probs      = st.session_state.get("last_probs")
    validation = st.session_state.get("last_validation")

    # Nothing analysed yet — nothing to show
    if img is None:
        return

    H('<hr class="cv-hr">')
    # ── Cassava Validity Banner ───────────────────────────────────────────────
    if validation and validation.get("method") == "isolation_forest":
        vp    = validation["validity_percentage"]
        # Colour the percentage based on how confident the gate is
        v_col = "#2E7D32" if vp >= 60 else ("#FF8F00" if vp >= 35 else "#C62828")

        if validation["is_cassava"]:
            H(f"""
            <div class="valid-banner-ok">
              <div class="valid-icon">✓</div>
              <div>
                <div class="valid-title">Cassava leaf detected</div>
                <div class="valid-body">
                  <span class="validity-score-badge">
                    Cassava Validity Score:&nbsp;
                    <span style="color:{v_col}">{vp:.1f}%</span>
                  </span><br>
                  Proceeding with disease analysis…
                </div>
              </div>
            </div>
            """)
        else:
            # Image rejected — display rejection message and stop.
            # Disease predictions, Grad-CAM, and recommendations must not render.
            H(f"""
            <div class="ood-banner">
              <div class="ood-icon">⚠️</div>
              <div>
                <div class="ood-title">
                  Uploaded image does not appear to contain a cassava leaf.
                </div>
                <div class="ood-body">
                  <span class="validity-score-badge">
                    Cassava Validity Score:&nbsp;
                    <span style="color:{v_col}">{vp:.1f}%</span>
                  </span><br><br>
                  Please upload a cassava leaf image for analysis.
                </div>
              </div>
            </div>
            """)
            return  # ← hard stop: no disease output below this line
    # ── Disease Prediction Results ────────────────────────────────────────────
    if result is None or probs is None:
        return

    pred_idx, ood = result

    if ood:
        H("""
        <div class="ood-banner">
          <div class="ood-icon">⚠️</div>
          <div>
            <div class="ood-title">Non-Cassava Image Detected — Diagnosis Flagged</div>
            <div class="ood-body">
              The model's prediction is highly uncertain for this image, suggesting it may not
              be a cassava leaf. This could be a non-leaf photo, a different crop, or a
              poorly-lit / blurry image.<br><br>
              Please upload a <strong>clear, well-lit cassava leaf photo</strong> where the
              leaf fills most of the frame for reliable diagnosis.
            </div>
          </div>
        </div>
        """)
    else:
        model_key = MODELS_META[display_name]["key"]
        tab1, tab2, tab3 = st.tabs([
            "📊 Results & Recommendations",
            "📈 Probability Charts",
            "🔥 Grad-CAM Explainability",
        ])
        with tab1:
            render_prediction_card(pred_idx, probs, st.session_state.get("crop_select", "Cassava"))
            render_confidence_section(probs, pred_idx)
            render_recommendations(pred_idx)
        with tab2:
            render_plotly_charts(probs, pred_idx)
        with tab3:
            render_gradcam_section(img, model, device, pred_idx, model_key)


# ═════════════════════════════════════════════════════════════════════════════
# UI — BATCH DISEASE SCREENING
# ═════════════════════════════════════════════════════════════════════════════

def render_batch_section(
    display_name: str, crop: str, model, device, model_ready: bool
) -> None:
    H('<div class="sec-label"><span class="sec-num">B</span>Batch Disease Screening</div>')

    H("""
    <div class="upload-zone" style="border-style:solid">
      <div class="upload-zone-hdr">
        <div class="upload-icon">📦</div>
        <div class="upload-title">Upload Multiple Images or a ZIP Archive</div>
        <div class="upload-hint">
          Select many images at once · Or pack them into a ZIP · Max 500 MB per file
        </div>
        <div class="chip-row">
          <span class="chip">ZIP</span><span class="chip">JPG</span>
          <span class="chip">JPEG</span><span class="chip">PNG</span><span class="chip">WEBP</span>
        </div>
      </div>
    """)
    batch_files = st.file_uploader(
        "Batch upload",
        type=["zip", "jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key=f"batch_uploader_{st.session_state['batch_uploader_gen']}",
        label_visibility="collapsed",
    )
    H('</div>')

    if not batch_files:
        H("""
        <div class="tip" style="margin-top:.75rem">
          <div class="tip-icon">💡</div>
          <div>
            <div class="tip-title">How batch screening works</div>
            <div class="tip-body">
              Upload multiple images directly or a ZIP archive containing cassava leaf photos.
              The system runs inference on every image and produces a downloadable CSV/Excel report.
              Non-cassava images are automatically flagged.
            </div>
          </div>
        </div>
        """)
        return

    if not model_ready:
        H("""
        <div class="info-banner">
          ℹ️ No checkpoint found — batch results are <strong>simulated (demo mode)</strong>.
          Place <code>.pth</code> files in <code>models/</code> for real inference.
        </div>
        """)

    H(f'<div style="font-size:.82rem;color:var(--text-2);margin:.5rem 0">'
      f'{len(batch_files)} file(s) selected</div>')

    run_col, rst_col, _ = st.columns([2, 1, 2])
    with run_col:
        run = st.button("▶ Run Batch Analysis", key="btn_batch_run", use_container_width=True)
    with rst_col:
        if st.button("↺ Reset", key="btn_batch_reset", use_container_width=True):
            st.session_state["batch_results"] = None
            st.session_state["batch_uploader_gen"] += 1
            st.rerun()

    if run:
        with st.spinner(f"Processing with {display_name}…"):
            df = process_batch(batch_files, display_name, model, device, crop)
        if df is not None:
            st.session_state["batch_results"] = df
            st.session_state["images_processed"] = (
                st.session_state.get("images_processed", 0) + len(df)
            )
            st.toast(f"Batch processing completed — {len(df)} images!", icon="✅")

    df = st.session_state.get("batch_results")
    if df is None or not _PANDAS_OK:
        return
    # ── summary metrics ───────────────────────────────────────────────────────
    H('<hr class="cv-hr">')
    total    = len(df)
    healthy  = (df["Prediction"] == "Healthy").sum()
    diseased = total - healthy - (df["Prediction"] == "Non-cassava (flagged)").sum()
    flagged  = (df["Prediction"] == "Non-cassava (flagged)").sum()
    H(f"""
    <div class="stat-strip" style="margin-bottom:1rem">
      <div class="stat-card"><div class="stat-icon b">📷</div>
        <div><div class="stat-val">{total}</div><div class="stat-lbl">Total Images</div></div></div>
      <div class="stat-card"><div class="stat-icon g">✅</div>
        <div><div class="stat-val">{healthy}</div><div class="stat-lbl">Healthy</div></div></div>
      <div class="stat-card"><div class="stat-icon r">🦠</div>
        <div><div class="stat-val">{diseased}</div><div class="stat-lbl">Diseased</div></div></div>
      <div class="stat-card"><div class="stat-icon o">⚠️</div>
        <div><div class="stat-val">{flagged}</div><div class="stat-lbl">Non-cassava</div></div></div>
    </div>
    """)
    # ── results table ─────────────────────────────────────────────────────────
    H('<div class="sec-label">Results Table</div>')
    st.dataframe(df, use_container_width=True, hide_index=True)
    # ── download buttons ──────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dl1, dl2, _ = st.columns([1, 1, 2])
    with dl1:
        st.download_button(
            "⬇ Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"cassava_vision_batch_{ts}.csv",
            mime="text/csv", key="dl_csv", use_container_width=True,
        )
    with dl2:
        if _OPENPYXL_OK:
            buf = io.BytesIO()
            df.to_excel(buf, index=False, engine="openpyxl")
            st.download_button(
                "⬇ Download Excel",
                data=buf.getvalue(),
                file_name=f"cassava_vision_batch_{ts}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_excel", use_container_width=True,
            )
    # ── distribution chart ────────────────────────────────────────────────────
    if _PLOTLY_OK and len(df) > 0:
        H('<div class="sec-label" style="margin-top:1rem">Disease Distribution</div>')
        dist = df["Prediction"].value_counts().reset_index()
        dist.columns = ["Disease", "Count"]
        cmap = {
            "Healthy":                              "#2E7D32",
            "Cassava Bacterial Blight (CBB)":       "#C62828",
            "Cassava Brown Streak Disease (CBSD)":  "#AD1457",
            "Cassava Green Mottle (CGM)":           "#FF8F00",
            "Cassava Mosaic Disease (CMD)":         "#E65100",
            "Non-cassava (flagged)":                "#78909C",
        }
        fig = go.Figure(go.Bar(
            x=dist["Disease"], y=dist["Count"],
            marker_color=[cmap.get(d, "#546E7A") for d in dist["Disease"]],
            marker_line_width=0,
            text=dist["Count"], textposition="outside",
            hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
        ))
        fig.update_layout(
            height=280, margin=dict(l=10,r=10,t=30,b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, tickfont={"size":10,"family":"Inter"}),
            yaxis=dict(showgrid=True, gridcolor="#F0F0F0", title="Count"),
            title=dict(text="Batch Disease Distribution",
                       font={"size":13,"color":"#546E7A","family":"Inter"}),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ═════════════════════════════════════════════════════════════════════════════
# UI — SCANNING TIPS
# ═════════════════════════════════════════════════════════════════════════════

def render_tips() -> None:
    H('<div class="sec-label" style="margin-top:1.5rem"><span class="sec-num">i</span>Scanning Tips</div>')
    for icon, title, body in [
        ("🌞", "Good lighting",    "Natural daylight is best — avoid harsh shadows or direct flash."),
        ("🔍", "Frame the leaf",   "Fill the image with a single cassava leaf for best accuracy."),
        ("📐", "Both surfaces",    "Scan upper and lower leaf surfaces separately if unsure."),
        ("🔄", "Re-scan if low",   "Confidence below 45% means try a clearer angle or better lighting."),
        ("📦", "Batch screening",  "Upload multiple images or a ZIP to analyse an entire field row."),
    ]:
        H(f"""
        <div class="tip">
          <div class="tip-icon">{icon}</div>
          <div><div class="tip-title">{title}</div><div class="tip-body">{body}</div></div>
        </div>
        """)


# ═════════════════════════════════════════════════════════════════════════════
# UI — FOOTER
# ═════════════════════════════════════════════════════════════════════════════

def render_footer() -> None:
    H("""
    <div class="cv-footer">
      <div style="font-size:1.5rem;margin-bottom:.5rem">🌿</div>
      <div style="font-size:1rem;font-weight:700;color:var(--text)">Cassava Vision AI</div>
      <div style="margin-top:4px;color:var(--text-2)">
        AI-Powered Crop Disease Detection &amp; Explainable Diagnostics
      </div>
      <div style="margin-top:10px;color:var(--text-2)">
        Developed for Agricultural Research and Decision Support
      </div>
      <div style="margin-top:6px;color:var(--text-3)">
        Powered by <strong>Streamlit</strong> · <strong>PyTorch</strong> ·
        <strong>OpenCV</strong> · <strong>Plotly</strong>
      </div>
      <div style="margin-top:8px;font-size:.7rem;color:var(--text-3)">
        &copy; 2026 AgroVision Africa. All Rights Reserved.
      </div>
    </div>
    """)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    init_session_state()

    # Sidebar — returns selected model display name, crop, model status
    display_name, crop, model_ready, model, device = render_sidebar()

    # Hero + stats
    render_header()
    render_stat_strip(model_ready)

    # Main tabs
    tab_single, tab_batch = st.tabs([
        "🔬 Single Image Analysis",
        "📦 Batch Disease Screening",
    ])
    with tab_single:
        render_single_analysis(display_name, crop, model, device, model_ready)
    with tab_batch:
        render_batch_section(display_name, crop, model, device, model_ready)

    render_tips()
    render_footer()


if __name__ == "__main__":
    main()
