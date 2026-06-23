"""
AgroVision Africa — Streamlit App (Enhanced UI)

Upload or capture cassava leaf images to get disease predictions with
class probabilities, Grad-CAM explanations, treatment advice, and
per-class confidence breakdowns.

Run:  streamlit run demo/app.py   (from project root)
      OR
      streamlit run app.py        (from inside demo/)
"""

from __future__ import annotations

import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import requests as _requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

# ── path resolution ───────────────────────────────────────────────────────────
FILE = Path(__file__).resolve()
DEMO_DIR = FILE.parent
ROOT = DEMO_DIR
for _candidate in [DEMO_DIR, DEMO_DIR.parent]:
    if (_candidate / "codes").exists():
        ROOT = _candidate
        break
sys.path.insert(0, str(ROOT))

# ── project imports ───────────────────────────────────────────────────────────
try:
    from codes.config import NUM_CLASSES, INPUT_SIZE, MODELS_DIR
    from codes.data_handler import CLASS_NAMES, CLASS_DESCRIPTIONS, get_transforms
    from codes.model import build_model
    from codes.utils import get_device
    from codes.gradcam import GradCAM, overlay_heatmap, preprocess_for_gradcam
    _CODES_OK = True
except ModuleNotFoundError:
    _CODES_OK = False
    NUM_CLASSES = 5
    INPUT_SIZE = 224
    MODELS_DIR = ROOT / "models"
    CLASS_NAMES = {
        0: "Cassava Bacterial Blight (CBB)",
        1: "Cassava Brown Streak Disease (CBSD)",
        2: "Cassava Green Mottle (CGM)",
        3: "Cassava Mosaic Disease (CMD)",
        4: "Healthy",
    }
    CLASS_DESCRIPTIONS = {
        0: "Bacterial infection causing angular leaf spots and wilting.",
        1: "Viral disease with yellow/brown streaks along leaf veins.",
        2: "Viral disease with mottled green patches on leaves.",
        3: "Viral disease causing mosaic patterns and leaf distortion.",
        4: "No visible disease symptoms detected.",
    }

# ── constants ─────────────────────────────────────────────────────────────────
TREATMENTS: dict[int, str] = {
    0: "Remove and destroy infected plant material. Apply copper-based bactericide. Source disease-free cuttings for replanting.",
    1: "Rogue out infected plants immediately — no cure exists. Plant CBSD-tolerant varieties (e.g. Narocass 1). Control whitefly populations.",
    2: "Monitor closely; CGM is usually mild. Maintain healthy soil nutrition. Source certified virus-free planting material.",
    3: "Uproot severely infected plants. Plant CMD-resistant varieties (e.g. TMEB419, Serere). Eliminate whitefly vectors with neem-based spray.",
    4: "No action required. Continue regular crop scouting every 2 weeks. Maintain good soil health and weed management.",
}

RECOMMENDATION_STEPS: dict[int, list[dict]] = {
    0: [
        {"icon": "🔴", "step": "Isolate immediately", "detail": "Mark and quarantine affected plants to prevent spread to neighbours."},
        {"icon": "🪣", "step": "Apply bactericide", "detail": "Spray with copper-based bactericide (e.g. Kocide 3000) covering leaf undersides."},
        {"icon": "✂️", "step": "Remove infected tissue", "detail": "Cut and burn all infected leaves, stems, and roots. Sterilise tools between plants."},
        {"icon": "🌱", "step": "Replant clean cuttings", "detail": "Source certified disease-free cuttings from a reputable nursery before replanting."},
    ],
    1: [
        {"icon": "🚨", "step": "Rogue out now", "detail": "Uproot and burn the entire infected plant — no chemical cure exists for CBSD."},
        {"icon": "🦟", "step": "Control whiteflies", "detail": "Apply imidacloprid or neem-based spray to reduce whitefly vector populations."},
        {"icon": "🌾", "step": "Plant tolerant varieties", "detail": "Restock with CBSD-tolerant varieties: Narocass 1, NASE 14, or local certified stock."},
        {"icon": "🔍", "step": "Scout weekly", "detail": "Inspect surrounding plants every 7 days for early CBSD symptoms."},
    ],
    2: [
        {"icon": "📋", "step": "Monitor closely", "detail": "CGM is generally mild — tag the plant and check for symptom progression weekly."},
        {"icon": "🌿", "step": "Boost soil health", "detail": "Apply balanced NPK fertiliser and organic compost to improve plant resilience."},
        {"icon": "🌱", "step": "Use certified planting material", "detail": "For next season, source virus-tested cuttings to prevent CGM introduction."},
        {"icon": "🦟", "step": "Reduce vector pressure", "detail": "Apply neem-based foliar spray to suppress mealybug and whitefly populations."},
    ],
    3: [
        {"icon": "🚨", "step": "Uproot severe cases", "detail": "Remove and burn plants showing more than 50% leaf distortion or stunting."},
        {"icon": "🌾", "step": "Plant CMD-resistant varieties", "detail": "Restock with TMEB419, Serere, or other CMD-resistant certified varieties."},
        {"icon": "🦟", "step": "Eliminate whitefly vectors", "detail": "Spray neem oil or imidacloprid fortnightly during the dry season."},
        {"icon": "🔍", "step": "Scout every 2 weeks", "detail": "Early detection prevents field-wide CMD spread. Use yellow sticky traps."},
    ],
    4: [
        {"icon": "✅", "step": "No treatment needed", "detail": "The plant shows no disease symptoms — continue normal management practices."},
        {"icon": "🔍", "step": "Scout every 2 weeks", "detail": "Routine field inspections catch diseases before they spread significantly."},
        {"icon": "🌿", "step": "Maintain soil health", "detail": "Apply compost and balanced fertiliser to keep plants vigorous and disease-resistant."},
        {"icon": "🌱", "step": "Manage weeds", "detail": "Remove weeds that harbour whiteflies and other vectors near cassava beds."},
    ],
}

SEVERITY: dict[int, str]       = {0: "High", 1: "High", 2: "Medium", 3: "High", 4: "None"}
SEVERITY_COLOR: dict[str, str] = {"High": "#ef4444", "Medium": "#f59e0b", "None": "#22c55e"}
CLASS_SHORT: dict[int, str]    = {0: "CBB", 1: "CBSD", 2: "CGM", 3: "CMD", 4: "Healthy"}

MODELS_META: dict[str, dict] = {
    "cassava_cnn": {
        "label": "CassavaCNN",
        "desc":  "Custom from-scratch CNN with residual blocks and SE attention.",
        "tag":   "CNN · custom",
    },
    "efficientnet_v2_s": {
        "label": "EfficientNet-V2-S",
        "desc":  "Fused-MBConv blocks — best accuracy-to-compute ratio.",
        "tag":   "CNN · recommended",
    },
    "swin_tiny": {
        "label": "Swin-Tiny",
        "desc":  "Vision Transformer with local-window attention and LoRA fine-tuning.",
        "tag":   "ViT · LoRA-adapted",
    },
}

OOD_MAX_PROB     = 0.45
OOD_ENTROPY      = 0.82
LEAF_GREEN_RATIO = 0.10

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgroVision Africa",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

:root {
  --bg:        #060d08;
  --surface:   #0b160d;
  --panel:     #101c12;
  --panel2:    #142016;
  --border:    #1b2e1e;
  --border2:   #243828;
  --green:     #2dda6e;
  --green-dim: #2dda6e22;
  --green-mid: #2dda6e44;
  --teal:      #00e5c8;
  --amber:     #ffb347;
  --red:       #ff5252;
  --sky:       #60c8ff;
  --text:      #e8f5eb;
  --text2:     #7daa88;
  --text3:     #3a5c42;
  --mono:      'Space Mono', monospace;
  --sans:      'Space Grotesk', sans-serif;
  --radius:    16px;
  --radius-sm: 10px;
}

/* ── base ── */
.stApp { background: var(--bg); font-family: var(--sans); color: var(--text); }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.75rem 2.5rem 5rem; max-width: 1440px; }
h1,h2,h3,h4 { font-family: var(--sans); color: var(--text); }
* { box-sizing: border-box; }

/* ── hero header ── */
.hero {
  position: relative;
  overflow: hidden;
  background: linear-gradient(135deg, #0b1f0e 0%, #061008 60%, #030b05 100%);
  border: 1px solid var(--border2);
  border-radius: 20px;
  padding: 2.5rem 2.75rem;
  margin-bottom: 2rem;
}
.hero::before {
  content: '';
  position: absolute;
  top: -80px; right: -80px;
  width: 320px; height: 320px;
  background: radial-gradient(circle, #2dda6e18 0%, transparent 70%);
  border-radius: 50%;
  pointer-events: none;
}
.hero::after {
  content: '';
  position: absolute;
  bottom: -60px; left: 30%;
  width: 200px; height: 200px;
  background: radial-gradient(circle, #00e5c810 0%, transparent 70%);
  border-radius: 50%;
  pointer-events: none;
}
.hero-inner {
  display: flex;
  align-items: center;
  justify-content: space-between;
  position: relative;
  z-index: 1;
}
.hero-left { display: flex; align-items: center; gap: 20px; }
.hero-emblem {
  width: 60px; height: 60px; border-radius: 14px;
  background: linear-gradient(135deg, #2dda6e, #00e5c8);
  display: flex; align-items: center; justify-content: center;
  font-size: 28px; flex-shrink: 0;
  box-shadow: 0 0 30px #2dda6e30;
}
.hero-title {
  font-size: 1.55rem; font-weight: 700;
  color: var(--text); margin: 0; line-height: 1.1;
  letter-spacing: -0.02em;
}
.hero-sub {
  font-size: 0.75rem; color: var(--text3);
  letter-spacing: 0.12em; text-transform: uppercase;
  margin-top: 4px; font-family: var(--mono);
}
.hero-badge {
  background: var(--green-dim); border: 1px solid var(--green-mid);
  color: var(--green); font-family: var(--mono); font-size: 0.65rem;
  padding: 6px 14px; border-radius: 6px; letter-spacing: 0.08em;
  white-space: nowrap;
}
.hero-badge.offline {
  background: #ff525218; border-color: #ff525230; color: var(--red);
}

/* ── KPI strip ── */
.kpi-strip {
  display: grid; grid-template-columns: repeat(4,1fr);
  border: 1px solid var(--border); border-radius: var(--radius);
  overflow: hidden; margin-bottom: 2rem;
  background: var(--surface);
}
.kpi {
  padding: 1.1rem 1.35rem;
  border-right: 1px solid var(--border);
  position: relative;
}
.kpi:last-child { border-right: none; }
.kpi-v {
  font-family: var(--mono); font-size: 1.5rem;
  font-weight: 700; color: var(--text); line-height: 1;
}
.kpi-v.g { color: var(--green); }
.kpi-v.t { color: var(--teal); }
.kpi-v.a { color: var(--amber); }
.kpi-l { font-size: 0.68rem; color: var(--text3); margin-top: 5px; text-transform: uppercase; letter-spacing: 0.08em; }

/* ── section label ── */
.sec-label {
  font-family: var(--mono); font-size: 0.68rem; font-weight: 700;
  letter-spacing: 0.14em; text-transform: uppercase; color: var(--text2);
  margin: 0 0 1rem;
  display: flex; align-items: center; gap: 8px;
}
.sec-num {
  background: var(--green-dim); color: var(--green);
  font-size: 0.6rem; width: 20px; height: 20px;
  display: flex; align-items: center; justify-content: center;
  border-radius: 4px; flex-shrink: 0; font-weight: 700;
}

/* ── input mode toggle ── */
.mode-toggle {
  display: grid; grid-template-columns: 1fr 1fr;
  gap: 0; border: 1px solid var(--border2);
  border-radius: var(--radius-sm); overflow: hidden;
  margin-bottom: 1rem;
}
.mode-btn {
  padding: 0.75rem 1rem;
  text-align: center;
  font-size: 0.82rem; font-weight: 600;
  cursor: pointer; transition: all 0.15s;
  background: var(--surface); color: var(--text2);
  border: none; outline: none;
  font-family: var(--sans);
}
.mode-btn:first-child { border-right: 1px solid var(--border2); }
.mode-btn.active {
  background: var(--green-dim); color: var(--green);
}
.mode-btn:hover:not(.active) { background: var(--panel2); color: var(--text); }

/* ── upload zone ── */
.upload-zone-wrapper {
  border: 2px dashed var(--border2);
  border-radius: var(--radius);
  background: linear-gradient(135deg, var(--panel) 0%, var(--surface) 100%);
  position: relative;
  overflow: hidden;
  transition: border-color 0.2s, background 0.2s;
}
.upload-zone-wrapper:hover {
  border-color: var(--green);
  background: linear-gradient(135deg, var(--panel2) 0%, #0d1e0f 100%);
}
.upload-zone-header {
  padding: 2rem 1.5rem 0.75rem;
  text-align: center;
  pointer-events: none;
}
.upload-zone-icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
.upload-zone-title {
  font-size: 0.95rem; font-weight: 600; color: var(--text);
  margin-bottom: 4px;
}
.upload-zone-hint {
  font-size: 0.73rem; color: var(--text3); font-family: var(--mono);
  letter-spacing: 0.06em;
}
.upload-zone-formats {
  display: flex; justify-content: center; gap: 6px; margin-top: 8px;
}
.format-pill {
  background: var(--border); color: var(--text2);
  font-family: var(--mono); font-size: 0.6rem;
  padding: 2px 8px; border-radius: 4px;
}

/* ── camera zone ── */
.camera-zone-wrapper {
  border: 2px solid #00e5c830;
  border-radius: var(--radius);
  background: linear-gradient(135deg, #030f0e 0%, var(--surface) 100%);
  overflow: hidden;
  position: relative;
}
.camera-zone-header {
  padding: 1.25rem 1.5rem 0.5rem;
  border-bottom: 1px solid #00e5c820;
  display: flex; align-items: center; gap: 10px;
}
.camera-pulse {
  width: 10px; height: 10px; border-radius: 50%;
  background: var(--teal);
  box-shadow: 0 0 8px var(--teal);
  animation: pulse 2s infinite;
}
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}
.camera-zone-title {
  font-size: 0.82rem; font-weight: 600; color: var(--teal);
  font-family: var(--mono); letter-spacing: 0.08em;
}
.camera-zone-body { padding: 0.75rem 1rem 1rem; }

/* ── native Streamlit file uploader overrides ── */
div[data-testid="stFileUploaderDropzone"] {
  border: none !important;
  border-radius: 0 !important;
  padding: 0.5rem 1.5rem 1.25rem !important;
  background: transparent !important;
  cursor: pointer !important;
}
div[data-testid="stFileUploader"] > label { display: none !important; }
div[data-testid="stFileUploader"] section {
  background: transparent !important; border: none !important; padding: 0 !important;
}
div[data-testid="stFileUploaderDropzoneInstructions"] {
  color: var(--text3) !important;
}
/* Camera input overrides */
div[data-testid="stCameraInput"] {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
}
div[data-testid="stCameraInput"] video { border-radius: var(--radius-sm) !important; }
div[data-testid="stCameraInput"] button {
  background: #00e5c815 !important; border: 1px solid #00e5c840 !important;
  color: var(--teal) !important; border-radius: 8px !important; width: 100% !important;
  margin-top: 8px !important; font-family: var(--sans) !important;
}

/* ── prediction card ── */
.pred-card {
  background: var(--panel);
  border: 1px solid var(--border2);
  border-radius: var(--radius); padding: 1.5rem;
  margin-top: 1rem;
  position: relative; overflow: hidden;
}
.pred-card::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 3px;
}
.pred-card.sev-high::before { background: linear-gradient(90deg, var(--red), transparent); }
.pred-card.sev-med::before  { background: linear-gradient(90deg, var(--amber), transparent); }
.pred-card.sev-none::before { background: linear-gradient(90deg, var(--green), transparent); }

.pred-eyebrow {
  font-family: var(--mono); font-size: 0.58rem;
  letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--text3); margin-bottom: 5px;
}
.pred-name { font-size: 1.2rem; font-weight: 700; color: var(--text); line-height: 1.2; }
.pred-desc {
  font-size: 0.8rem; color: var(--text2); line-height: 1.7;
  margin: 0.85rem 0 1rem;
}

/* ── confidence pill ── */
.conf-pill {
  padding: 5px 14px; border-radius: 20px;
  font-family: var(--mono); font-size: 0.73rem; font-weight: 700;
  white-space: nowrap; display: inline-flex; align-items: center; gap: 6px;
}
.conf-dot { width: 6px; height: 6px; border-radius: 50%; }

/* ── probability bars ── */
.prob-section { margin-top: 0.5rem; }
.prob-header {
  font-family: var(--mono); font-size: 0.58rem;
  letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--text3); margin-bottom: 0.75rem;
}
.bar-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
.bar-lbl {
  font-size: 0.7rem; color: var(--text2);
  width: 50px; flex-shrink: 0; font-family: var(--mono);
}
.bar-lbl.top { color: var(--text); font-weight: 700; }
.bar-track {
  flex: 1; height: 6px; background: var(--border);
  border-radius: 4px; overflow: hidden;
  position: relative;
}
.bar-fill {
  height: 100%; border-radius: 4px;
  transition: width 0.6s cubic-bezier(0.4,0,0.2,1);
}
.bar-pct {
  font-family: var(--mono); font-size: 0.67rem;
  color: var(--text3); width: 38px; text-align: right; flex-shrink: 0;
}
.bar-pct.top { color: var(--text); font-weight: 700; }

/* ── confidence gauge ── */
.gauge-wrap {
  display: flex; align-items: center; gap: 12px;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius-sm); padding: 0.85rem 1rem;
  margin: 0.75rem 0;
}
.gauge-bar-outer {
  flex: 1; height: 10px; background: var(--border);
  border-radius: 6px; overflow: hidden;
}
.gauge-bar-inner {
  height: 100%; border-radius: 6px;
  background: linear-gradient(90deg, #ef4444 0%, #f59e0b 45%, #2dda6e 75%, #00e5c8 100%);
  position: relative;
}
.gauge-marker {
  position: absolute; top: -4px; height: 18px; width: 3px;
  background: white; border-radius: 2px;
  box-shadow: 0 0 6px rgba(255,255,255,0.8);
  transform: translateX(-50%);
}
.gauge-label {
  font-family: var(--mono); font-size: 0.7rem; color: var(--text2);
  white-space: nowrap; flex-shrink: 0;
}
.gauge-value {
  font-family: var(--mono); font-size: 1.1rem; font-weight: 700;
  white-space: nowrap; flex-shrink: 0;
}

/* ── recommendation steps ── */
.rec-section {
  margin-top: 1rem;
}
.rec-header {
  display: flex; align-items: center; gap: 8px;
  margin-bottom: 0.75rem;
}
.rec-title {
  font-family: var(--mono); font-size: 0.62rem;
  letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--green); font-weight: 700;
}
.rec-steps { display: flex; flex-direction: column; gap: 6px; }
.rec-step {
  display: flex; align-items: flex-start; gap: 10px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-left: 3px solid var(--green);
  border-radius: var(--radius-sm); padding: 0.75rem 0.9rem;
  position: relative; overflow: hidden;
}
.rec-step.urgent { border-left-color: var(--red); }
.rec-step.warn   { border-left-color: var(--amber); }
.rec-step.ok     { border-left-color: var(--green); }
.rec-step-icon { font-size: 1.1rem; flex-shrink: 0; margin-top: 1px; }
.rec-step-body {}
.rec-step-title {
  font-size: 0.8rem; font-weight: 600; color: var(--text); margin-bottom: 2px;
}
.rec-step-detail {
  font-size: 0.73rem; color: var(--text2); line-height: 1.55;
}
.rec-step-num {
  position: absolute; top: 6px; right: 8px;
  font-family: var(--mono); font-size: 0.6rem;
  color: var(--text3); opacity: 0.5;
}

/* ── severity badge ── */
.sev-badge { display: inline-flex; align-items: center; gap: 5px; margin-top: 6px; }
.sev-dot { width: 7px; height: 7px; border-radius: 50%; }
.sev-txt { font-family: var(--mono); font-size: 0.62rem; }

/* ── batch card ── */
.batch-card {
  background: var(--panel); border: 1px solid var(--border2);
  border-radius: var(--radius-sm); padding: 0.65rem;
  margin-top: 0.5rem; text-align: center;
}
.batch-label { font-size: 0.8rem; font-weight: 600; color: var(--text); margin-bottom: 4px; }
.batch-conf  { font-family: var(--mono); font-size: 0.72rem; }
.batch-sev   { font-family: var(--mono); font-size: 0.6rem; margin-top: 3px; }

/* ── batch confidence bar (mini) ── */
.mini-bar-track {
  width: 100%; height: 4px; background: var(--border);
  border-radius: 3px; overflow: hidden; margin-top: 6px;
}
.mini-bar-fill { height: 100%; border-radius: 3px; }

/* ── OOD banner ── */
.ood-banner {
  background: #120800; border: 1px solid #ffb34730;
  border-left: 3px solid var(--amber);
  border-radius: var(--radius-sm); padding: 1.1rem 1.25rem;
  display: flex; align-items: flex-start; gap: 14px;
  margin-top: 1rem;
}
.ood-icon  { font-size: 1.6rem; flex-shrink: 0; }
.ood-title { font-weight: 700; color: var(--amber); margin-bottom: 4px; font-size: 0.9rem; }
.ood-body  { font-size: 0.78rem; color: var(--text2); line-height: 1.65; }

/* ── tips ── */
.tip {
  display: flex; align-items: flex-start; gap: 12px;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius-sm); padding: 11px 13px; margin-bottom: 8px;
}
.tip-icon { font-size: 1.1rem; flex-shrink: 0; margin-top: 1px; }
.tip-title { font-size: 0.78rem; font-weight: 600; color: var(--text); margin-bottom: 2px; }
.tip-body  { font-size: 0.72rem; color: var(--text2); line-height: 1.5; }

/* ── model selector ── */
.model-info-card {
  background: var(--surface); border: 1px solid var(--border2);
  border-left: 3px solid var(--green);
  border-radius: var(--radius-sm); padding: 0.8rem 1rem;
  margin-bottom: 1.25rem;
}
.model-tag {
  font-family: var(--mono); font-size: 0.58rem; color: var(--green);
  letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 4px;
}
.model-desc { font-size: 0.82rem; color: var(--text2); line-height: 1.5; }

/* ── no-model banner ── */
.no-model {
  background: var(--surface); border: 1px solid var(--border2);
  border-left: 3px solid var(--amber);
  border-radius: var(--radius-sm); padding: 1.25rem;
  font-size: 0.82rem; color: var(--text2); line-height: 1.7;
}

/* ── confidence legend ── */
.legend { display: flex; gap: 14px; flex-wrap: wrap; margin-top: 0.75rem; }
.leg-item { display: flex; align-items: center; gap: 5px; font-size: 0.68rem; color: var(--text3); }
.leg-dot  { width: 8px; height: 8px; border-radius: 2px; }

/* ── model card (sidebar) ── */
.mc-row { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px; }
.mc-k { font-size: 0.73rem; color: var(--text3); }
.mc-v { font-size: 0.73rem; color: var(--text); font-family: var(--mono); text-align: right; max-width: 55%; word-break: break-all; }

/* ── divider ── */
.divider { border: none; border-top: 1px solid var(--border); margin: 1rem 0; }

/* ── Streamlit overrides ── */
.stButton > button {
  background: var(--surface) !important; border: 1px solid var(--border2) !important;
  color: var(--text2) !important; border-radius: 8px !important;
  font-family: var(--sans) !important; font-size: 0.84rem !important;
  width: 100%; padding: 0.55rem 1rem !important;
  transition: all 0.15s !important;
}
.stButton > button:hover {
  border-color: var(--green) !important; color: var(--green) !important;
  background: var(--green-dim) !important;
}
.stTabs [data-baseweb="tab-list"] {
  background: transparent !important; border-bottom: 1px solid var(--border) !important;
  gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important; color: var(--text3) !important;
  font-family: var(--sans) !important; font-size: 0.84rem !important;
  border-bottom: 2px solid transparent !important; border-radius: 0 !important;
  padding: 0.5rem 0.75rem !important;
}
.stTabs [aria-selected="true"] {
  color: var(--green) !important; border-bottom-color: var(--green) !important;
  background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.25rem !important; }
div[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }
div[data-testid="stSidebar"] * { color: var(--text2) !important; }
div[data-testid="stSelectbox"] > div > div {
  background: var(--surface) !important; border-color: var(--border2) !important; color: var(--text) !important;
}
div[data-testid="stRadio"] > div { gap: 8px !important; }
div[data-testid="stRadio"] label { color: var(--text2) !important; font-size: 0.88rem !important; }
.stImage img { border-radius: var(--radius-sm); }
div[data-testid="stExpander"] { background: var(--panel) !important; border-color: var(--border) !important; border-radius: var(--radius-sm) !important; }
div[data-testid="stExpander"] summary { font-size: 0.88rem !important; }
</style>
""", unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────────────────────────
def H(html: str) -> None:
    st.markdown(html, unsafe_allow_html=True)


def conf_color(p: float) -> str:
    if p >= 0.75: return "#2dda6e"
    if p >= 0.45: return "#ffb347"
    return "#ff5252"


def conf_label(p: float) -> str:
    if p >= 0.75: return "High confidence"
    if p >= 0.45: return "Medium confidence"
    return "Low — verify manually"


def sev_to_css(sev: str) -> str:
    return {"High": "sev-high", "Medium": "sev-med", "None": "sev-none"}.get(sev, "sev-none")


def step_urgency(pred_idx: int, step_i: int) -> str:
    if pred_idx == 4: return "ok"
    if pred_idx == 2: return "warn" if step_i == 0 else "ok"
    return "urgent" if step_i < 2 else "warn"


# ── checkpoint helpers ────────────────────────────────────────────────────────
def find_checkpoint(model_name: str) -> Path | None:
    base = Path(MODELS_DIR)
    candidates = [
        base / f"{model_name}_best_model.pth",
        base / model_name / "best_model.pth",
        base / "best_model.pth",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


# ── zip / google drive helpers ────────────────────────────────────────────────

def _extract_images_from_zip(zip_bytes: bytes | io.BytesIO) -> list[Image.Image]:
    images: list[Image.Image] = []
    buf = io.BytesIO(zip_bytes) if isinstance(zip_bytes, bytes) else zip_bytes
    try:
        with zipfile.ZipFile(buf) as zf:
            for name in zf.namelist():
                if Path(name).suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    try:
                        images.append(Image.open(io.BytesIO(zf.read(name))).convert("RGB"))
                    except Exception:
                        pass
    except zipfile.BadZipFile:
        pass
    return images


def _gdrive_file_id(url: str) -> str | None:
    for pattern in [r"/file/d/([a-zA-Z0-9_-]+)", r"[?&]id=([a-zA-Z0-9_-]+)"]:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    return None


def _download_gdrive(file_id: str) -> bytes | None:
    if not _REQUESTS_OK:
        return None
    try:
        session = _requests.Session()
        base_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        resp = session.get(base_url, stream=True, timeout=30)
        for k, v in resp.cookies.items():
            if k.startswith("download_warning"):
                resp = session.get(base_url + f"&confirm={v}", stream=True, timeout=30)
                break
        return resp.content if resp.status_code == 200 and len(resp.content) > 512 else None
    except Exception:
        return None


def _images_from_gdrive_url(url: str) -> tuple[list[Image.Image], str]:
    file_id = _gdrive_file_id(url.strip())
    if not file_id:
        return [], "Could not find a file ID in that URL. Paste a Google Drive share link."
    if not _REQUESTS_OK:
        return [], "The `requests` library is not installed. Run: pip install requests"
    data = _download_gdrive(file_id)
    if not data:
        return [], "Download failed. Make sure the file is shared with 'Anyone with the link'."
    if data[:2] == b"PK":  # ZIP magic bytes
        imgs = _extract_images_from_zip(data)
        return (imgs, "") if imgs else ([], "Zip file contained no valid JPG/PNG images.")
    try:
        return [Image.open(io.BytesIO(data)).convert("RGB")], ""
    except Exception:
        return [], "Could not open the downloaded file as an image. Expected JPG, PNG, or ZIP."


# ── model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model weights…")
def load_model(model_name: str, ckpt_path: str):
    device = get_device() if _CODES_OK else torch.device("cpu")
    if not _CODES_OK:
        return None, device
    try:
        model = build_model(model_name, num_classes=NUM_CLASSES, dropout=0.3)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        if isinstance(state, dict) and "net" in state:
            state = state["net"]
        model.load_state_dict(state, strict=False)
        model.eval().to(device)
        return model, device
    except Exception as e:
        st.error(f"Model load failed: {e}")
        return None, device


# ── inference ─────────────────────────────────────────────────────────────────
def run_inference(image: Image.Image, model, device) -> tuple[int, np.ndarray]:
    if not _CODES_OK or model is None:
        probs = np.random.dirichlet(np.ones(NUM_CLASSES))
        return int(np.argmax(probs)), probs
    tfm = get_transforms(INPUT_SIZE, augment=False)
    tensor = tfm(image=np.array(image.convert("RGB")))["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    return int(np.argmax(probs)), probs


def _has_leaf_colors(image: Image.Image) -> bool:
    arr = np.array(image.convert("RGB")).reshape(-1, 3).astype(float)
    r, g, b = arr[:, 0], arr[:, 1], arr[:, 2]
    leaf_mask = (g > r) & (g > b) & (g > 60) & (r < 180)
    return float(leaf_mask.sum() / leaf_mask.size) >= LEAF_GREEN_RATIO


def check_ood(probs: np.ndarray, image: Image.Image | None = None) -> bool:
    if image is not None and not _has_leaf_colors(image):
        return True
    max_prob = float(np.max(probs))
    n = len(probs)
    entropy = -float(np.sum(probs * np.log(probs + 1e-12))) / np.log(n)
    return max_prob < OOD_MAX_PROB or entropy > OOD_ENTROPY


def run_gradcam(image, model, device, pred_idx, model_name):
    if not _CODES_OK:
        return None, None
    try:
        img_np = np.array(image.convert("RGB"))
        input_tensor, resized_rgb = preprocess_for_gradcam(img_np, INPUT_SIZE)
        cam = GradCAM(model, model_name=model_name)
        heatmap, _ = cam(input_tensor.to(device), class_idx=pred_idx)
        cam.remove_hooks()
        overlay = overlay_heatmap(resized_rgb, heatmap)
        return overlay, resized_rgb
    except Exception:
        return None, None


# ── UI components ─────────────────────────────────────────────────────────────

def render_header(model_ready: bool, model_label: str):
    status = "MODEL READY" if model_ready else "NO CHECKPOINT"
    badge_cls = "hero-badge" if model_ready else "hero-badge offline"
    H(f"""
    <div class="hero">
      <div class="hero-inner">
        <div class="hero-left">
          <div class="hero-emblem">🌿</div>
          <div>
            <div class="hero-title">AgroVision Africa</div>
            <div class="hero-sub">Cassava Leaf Disease Classifier · {model_label}</div>
          </div>
        </div>
        <span class="{badge_cls}">{status}</span>
      </div>
    </div>
    """)


def render_kpis():
    H("""
    <div class="kpi-strip">
      <div class="kpi"><div class="kpi-v g">5</div><div class="kpi-l">Disease classes</div></div>
      <div class="kpi"><div class="kpi-v t">21,397</div><div class="kpi-l">Training images</div></div>
      <div class="kpi"><div class="kpi-v a">224 px</div><div class="kpi-l">Input resolution</div></div>
      <div class="kpi"><div class="kpi-v">ImageNet</div><div class="kpi-l">Pretrained backbone</div></div>
    </div>
    """)


def render_model_selector() -> str:
    H('<div class="sec-label"><span class="sec-num">1</span>Select Model</div>')
    options = list(MODELS_META.keys())
    labels  = [MODELS_META[k]["label"] for k in options]
    chosen_label = st.radio(
        "Model", labels, index=1, key="model_radio", label_visibility="collapsed",
    )
    selected_key = options[labels.index(chosen_label)]
    meta = MODELS_META[selected_key]
    H(f"""
    <div class="model-info-card">
      <div class="model-tag">{meta['tag']}</div>
      <div class="model-desc">{meta['desc']}</div>
    </div>
    """)
    return selected_key


def render_no_model(model_name: str):
    ckpt_hint = f"{MODELS_DIR}/{model_name}/best_model.pth"
    H(f"""
    <div class="no-model">
      <strong style="color:var(--amber)">No checkpoint found for {MODELS_META[model_name]['label']}</strong><br><br>
      Expected: <code>{ckpt_hint}</code><br>
      Train the model in <code>notebooks/AgroVision_Africa.ipynb</code>, then restart.
    </div>
    """)


def render_ood_banner():
    H("""
    <div class="ood-banner">
      <div class="ood-icon">⚠️</div>
      <div>
        <div class="ood-title">Not a cassava leaf — no diagnosis produced</div>
        <div class="ood-body">
          This image doesn't appear to be a cassava leaf. The classifier only works on cassava
          leaves and will not give meaningful results for faces, other crops, or unclear photos.<br><br>
          Upload a clear, well-lit photo where the <strong style="color:var(--text)">cassava leaf fills most of the frame</strong>.
        </div>
      </div>
    </div>
    """)


def render_confidence_gauge(p: float):
    col = conf_color(p)
    label = conf_label(p)
    pct = p * 100
    H(f"""
    <div class="gauge-wrap">
      <span class="gauge-label">Confidence</span>
      <div class="gauge-bar-outer">
        <div class="gauge-bar-inner" style="width:100%;position:relative">
          <div class="gauge-marker" style="left:{pct:.1f}%"></div>
        </div>
      </div>
      <span class="gauge-value" style="color:{col}">{pct:.1f}%</span>
      <span class="gauge-label" style="color:{col}">· {label}</span>
    </div>
    """)


def render_recommendation_steps(pred_idx: int):
    steps = RECOMMENDATION_STEPS[pred_idx]
    steps_html = ""
    for i, s in enumerate(steps):
        urg = step_urgency(pred_idx, i)
        steps_html += f"""
        <div class="rec-step {urg}">
          <div class="rec-step-icon">{s['icon']}</div>
          <div class="rec-step-body">
            <div class="rec-step-title">{s['step']}</div>
            <div class="rec-step-detail">{s['detail']}</div>
          </div>
          <div class="rec-step-num">0{i+1}</div>
        </div>"""
    H(f"""
    <div class="rec-section">
      <div class="rec-header">
        <div class="rec-title">📋 Recommended Actions</div>
      </div>
      <div class="rec-steps">{steps_html}</div>
    </div>
    """)


def render_pred_card(pred_idx: int, probs: np.ndarray):
    top_p   = float(probs[pred_idx])
    col     = conf_color(top_p)
    sev     = SEVERITY[pred_idx]
    sev_col = SEVERITY_COLOR[sev]
    sev_css = sev_to_css(sev)

    H(f"""
    <div class="pred-card {sev_css}">
      <div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:0.5rem;flex-wrap:wrap;gap:0.5rem">
        <div>
          <div class="pred-eyebrow">Predicted Condition</div>
          <div class="pred-name">{CLASS_NAMES[pred_idx]}</div>
          <div class="sev-badge">
            <span class="sev-dot" style="background:{sev_col}"></span>
            <span class="sev-txt" style="color:{sev_col}">Severity: {sev}</span>
          </div>
        </div>
        <span class="conf-pill" style="background:{col}18;color:{col};border:1px solid {col}40">
          <span class="conf-dot" style="background:{col}"></span>
          {top_p*100:.1f}%
        </span>
      </div>
      <div class="pred-desc">{CLASS_DESCRIPTIONS[pred_idx]}</div>
    </div>
    """)

    # Confidence gauge
    render_confidence_gauge(top_p)

    # Legend
    H("""
    <div class="legend">
      <span class="leg-item"><span class="leg-dot" style="background:#2dda6e"></span>≥75% High</span>
      <span class="leg-item"><span class="leg-dot" style="background:#ffb347"></span>45–74% Medium</span>
      <span class="leg-item"><span class="leg-dot" style="background:#ff5252"></span>&lt;45% Low — re-scan</span>
    </div>
    """)

    # Recommendations
    render_recommendation_steps(pred_idx)


def render_gradcam_tab(image, model, device, pred_idx, model_name):
    if not _CODES_OK:
        st.info("Grad-CAM requires the codes/ module. Check your installation.")
        return
    with st.spinner("Computing Grad-CAM…"):
        overlay, resized_rgb = run_gradcam(image, model, device, pred_idx, model_name)
    if overlay is None:
        st.warning("Grad-CAM unavailable for this configuration.")
        return
    col1, col2 = st.columns(2)
    with col1:
        H('<div class="sec-label" style="margin-top:0">Original</div>')
        st.image(resized_rgb, use_container_width=True)
    with col2:
        H('<div class="sec-label" style="margin-top:0">Grad-CAM Overlay</div>')
        st.image(overlay, use_container_width=True)
    H(f"""
    <div style="margin-top:0.75rem;padding:0.9rem 1rem;background:var(--surface);
      border:1px solid var(--border2);border-left:3px solid var(--teal);
      border-radius:var(--radius-sm);font-size:0.79rem;color:var(--text2);line-height:1.7">
      <span style="font-family:var(--mono);font-size:0.58rem;letter-spacing:0.1em;
        text-transform:uppercase;color:var(--teal)">What am I looking at?</span><br>
      <strong style="color:var(--text)">Grad-CAM</strong> highlights the leaf regions that most
      influenced the prediction.
      <span style="color:#ef4444">Red/yellow</span> = high influence ·
      <span style="color:#3b82f6">blue</span> = low influence.
    </div>
    <div style="margin-top:0.5rem;font-family:var(--mono);font-size:0.63rem;
      color:var(--text3);letter-spacing:0.05em">
      Explaining → <strong style="color:var(--text2)">{CLASS_NAMES[pred_idx]}</strong>
    </div>
    """)


def render_batch_results(images: list, model, device, model_name: str):
    H(f'<div class="sec-label" style="margin-top:1rem">Batch Results — {len(images)} images</div>')
    n_cols = min(len(images), 3)
    cols = st.columns(n_cols)
    for i, img in enumerate(images):
        with cols[i % n_cols]:
            st.image(img, use_container_width=True)
            pred_idx, probs = run_inference(img, model, device)
            is_ood = check_ood(probs, img)
            if is_ood:
                H("""
                <div class="batch-card" style="border-left:3px solid var(--amber)">
                  <div class="batch-label" style="color:var(--amber)">Not cassava</div>
                  <div class="batch-conf" style="color:var(--text3)">OOD detected</div>
                  <div class="batch-sev" style="color:var(--text3)">–</div>
                </div>
                """)
            else:
                top_p = float(probs[pred_idx])
                col   = conf_color(top_p)
                sev   = SEVERITY[pred_idx]
                sev_col = SEVERITY_COLOR[sev]
                bar_w = max(top_p * 100, 2)
                H(f"""
                <div class="batch-card">
                  <div class="batch-label">{CLASS_SHORT[pred_idx]}</div>
                  <div class="batch-conf" style="color:{col}">{top_p*100:.1f}%</div>
                  <div class="batch-sev" style="color:{sev_col}">Severity: {sev}</div>
                  <div class="mini-bar-track">
                    <div class="mini-bar-fill" style="width:{bar_w:.1f}%;background:{col}"></div>
                  </div>
                </div>
                """)

    # Summary table
    H('<div class="sec-label" style="margin-top:1.5rem">Batch Summary</div>')
    results = []
    for i, img in enumerate(images):
        pred_idx, probs = run_inference(img, model, device)
        is_ood = check_ood(probs, img)
        results.append({
            "Image": f"Image {i+1}",
            "Diagnosis": "OOD" if is_ood else CLASS_NAMES[pred_idx],
            "Confidence": "—" if is_ood else f"{float(probs[pred_idx])*100:.1f}%",
            "Severity": "—" if is_ood else SEVERITY[pred_idx],
        })
    import pandas as pd
    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_scan_panel(model, device, model_ready: bool, model_name: str):
    H('<div class="sec-label"><span class="sec-num">2</span>Leaf Disease Scanner</div>')

    if not model_ready:
        render_no_model(model_name)
        return

    images = []

    # ── mode toggle ──
    if "input_mode" not in st.session_state:
        st.session_state["input_mode"] = "upload"

    if st.session_state["input_mode"] == "upload":
        _, cam_col = st.columns([4, 1])
        with cam_col:
            if st.button("📷  Camera", key="btn_camera_mode", use_container_width=True):
                st.session_state["input_mode"] = "camera"
                st.rerun()
    else:
        if st.button("← Back to Upload", key="btn_upload_mode"):
            st.session_state["input_mode"] = "upload"
            st.rerun()

    H('<div style="height:0.5rem"></div>')

    # ── upload zone ──
    if st.session_state["input_mode"] == "upload":
        H("""
        <div class="upload-zone-wrapper">
          <div class="upload-zone-header">
            <div class="upload-zone-icon">🌿</div>
            <div class="upload-zone-title">Drop Cassava Leaf Photos Here</div>
            <div class="upload-zone-hint">Single image · Batch · Zip archive · Drag &amp; drop or click to browse</div>
            <div class="upload-zone-formats">
              <span class="format-pill">JPG</span>
              <span class="format-pill">JPEG</span>
              <span class="format-pill">PNG</span>
              <span class="format-pill">ZIP</span>
            </div>
          </div>
        """)

        tab_local, tab_drive = st.tabs(["💻  Local Computer", "🔗  Google Drive"])

        with tab_local:
            uploaded = st.file_uploader(
                "Drop cassava leaf photos here",
                type=["jpg", "jpeg", "png", "zip"],
                accept_multiple_files=True,
                key="uploader",
                label_visibility="collapsed",
            )
            if uploaded:
                for f in uploaded:
                    if f.name.lower().endswith(".zip"):
                        extracted = _extract_images_from_zip(f.read())
                        if extracted:
                            images.extend(extracted)
                        else:
                            st.warning(f"No valid images found in {f.name}.")
                    else:
                        images.append(Image.open(f).convert("RGB"))

        with tab_drive:
            H("""
            <div style="font-size:0.78rem;color:var(--text3);margin-bottom:0.6rem;line-height:1.6">
              Paste a <strong style="color:var(--text2)">Google Drive shareable link</strong> for a JPG, PNG, or ZIP file.
              The file must be shared with <em>Anyone with the link</em>.
            </div>
            """)
            drive_url = st.text_input(
                "Google Drive link",
                placeholder="https://drive.google.com/file/d/…/view?usp=sharing",
                key="drive_url",
                label_visibility="collapsed",
            )
            if drive_url:
                with st.spinner("Downloading from Google Drive…"):
                    gdrive_images, gdrive_err = _images_from_gdrive_url(drive_url)
                if gdrive_err:
                    st.error(gdrive_err)
                else:
                    images.extend(gdrive_images)
                    st.success(f"Loaded {len(gdrive_images)} image(s) from Google Drive.")

        H("</div>")  # close upload-zone-wrapper

    # ── camera zone ──
    else:
        H("""
        <div class="camera-zone-wrapper">
          <div class="camera-zone-header">
            <div class="camera-pulse"></div>
            <div class="camera-zone-title">LIVE CAMERA · Point at a cassava leaf</div>
          </div>
          <div class="camera-zone-body">
        """)
        snap = st.camera_input(
            "camera",
            key="camera",
            label_visibility="collapsed",
        )
        H("</div></div>")  # close camera-zone-body + wrapper
        if snap:
            images = [Image.open(snap).convert("RGB")]

    if not images:
        return

    # ── single image flow ──
    if len(images) == 1:
        image = images[0]
        st.image(image, use_container_width=True, caption="Input image")
        with st.spinner("Analysing leaf…"):
            pred_idx, probs = run_inference(image, model, device)
            is_ood = check_ood(probs, image)

        if is_ood:
            render_ood_banner()
        else:
            tab_results, tab_gradcam = st.tabs(["📊  Results & Recommendations", "🔥  Grad-CAM Explanation"])
            with tab_results:
                render_pred_card(pred_idx, probs)
            with tab_gradcam:
                render_gradcam_tab(image, model, device, pred_idx, model_name)

    # ── batch flow ──
    else:
        H(f'<div style="margin:0.5rem 0;font-size:0.8rem;color:var(--text2)">{len(images)} images loaded — running batch analysis…</div>')
        with st.spinner(f"Analysing {len(images)} images…"):
            render_batch_results(images, model, device, model_name)

    H('<div style="height:0.5rem"></div>')
    if st.button("↻  Start new scan"):
        st.rerun()


def render_class_reference():
    H('<div class="sec-label" style="margin-top:1.75rem"><span class="sec-num">3</span>Disease Class Reference</div>')
    for i in range(NUM_CLASSES):
        sev_col = SEVERITY_COLOR[SEVERITY[i]]
        with st.expander(f"{CLASS_SHORT[i]} — {CLASS_NAMES[i]}"):
            steps = RECOMMENDATION_STEPS[i]
            steps_html = "".join(
                f'<div style="display:flex;gap:8px;margin-bottom:5px"><span>{s["icon"]}</span>'
                f'<div><span style="font-size:0.78rem;font-weight:600;color:var(--text)">{s["step"]}</span>'
                f'<div style="font-size:0.73rem;color:var(--text2)">{s["detail"]}</div></div></div>'
                for s in steps
            )
            H(f"""
            <div style="color:var(--text2);font-size:0.82rem;line-height:1.65;margin-bottom:0.75rem">
              {CLASS_DESCRIPTIONS[i]}
            </div>
            <div style="font-family:var(--mono);font-size:0.58rem;letter-spacing:0.1em;text-transform:uppercase;
              color:var(--green);margin-bottom:0.6rem">Recommended Actions</div>
            {steps_html}
            <div class="sev-badge" style="margin-top:10px">
              <span class="sev-dot" style="background:{sev_col}"></span>
              <span class="sev-txt" style="color:{sev_col}">Severity: {SEVERITY[i]}</span>
            </div>""")


def render_tips():
    H('<div class="sec-label" style="margin-top:1.5rem"><span class="sec-num">i</span>Scanning Tips</div>')
    tips = [
        ("🌞", "Good lighting",  "Natural daylight is best — avoid harsh shadows or direct flash."),
        ("🔍", "Frame the leaf", "Fill the image with a single leaf so the model can focus."),
        ("📐", "Both surfaces",  "If unsure, scan upper and lower leaf surfaces separately."),
        ("🔄", "Re-scan if low", "If confidence is below 45%, try a different angle or image."),
        ("📦", "Batch scanning", "Upload multiple images at once to analyse an entire field row."),
    ]
    for icon, title, body in tips:
        H(f"""
        <div class="tip">
          <div class="tip-icon">{icon}</div>
          <div>
            <div class="tip-title">{title}</div>
            <div class="tip-body">{body}</div>
          </div>
        </div>""")


def render_sidebar(selected_model: str, ckpt: Path | None):
    with st.sidebar:
        H('<div style="font-size:0.82rem;font-weight:700;color:var(--text);margin-bottom:0.85rem;font-family:var(--sans)">Model Card</div>')
        ckpt_str = str(ckpt) if ckpt else "— not found —"
        meta = MODELS_META[selected_model]
        H(f"""
        <div>
          <div class="mc-row"><span class="mc-k">Architecture</span><span class="mc-v">{meta['label']}</span></div>
          <div class="mc-row"><span class="mc-k">Classes</span><span class="mc-v">{NUM_CLASSES}</span></div>
          <div class="mc-row"><span class="mc-k">Input size</span><span class="mc-v">{INPUT_SIZE} × {INPUT_SIZE} px</span></div>
          <div class="mc-row"><span class="mc-k">Transfer</span><span class="mc-v">ImageNet → fine-tune</span></div>
          <div class="mc-row"><span class="mc-k">Dataset</span><span class="mc-v">Cassava Kaggle 2020</span></div>
          <div class="mc-row"><span class="mc-k">Checkpoint</span><span class="mc-v">{Path(ckpt_str).name if ckpt else 'not found'}</span></div>
        </div>
        <hr style="border-color:var(--border);margin:0.9rem 0">""")

        H('<div style="font-size:0.82rem;font-weight:700;color:var(--text);margin-bottom:0.7rem">Disease Classes</div>')
        for i in range(NUM_CLASSES):
            dot = SEVERITY_COLOR[SEVERITY[i]]
            H(f"""
            <div style="display:flex;align-items:center;gap:7px;margin-bottom:8px">
              <span style="width:7px;height:7px;border-radius:50%;background:{dot};display:inline-block;flex-shrink:0"></span>
              <span style="font-size:0.73rem"><b style="color:var(--text)">{CLASS_SHORT[i]}</b>
              &nbsp;<span style="color:var(--text2)">{CLASS_NAMES[i]}</span></span>
            </div>""")

        H('<hr style="border-color:var(--border);margin:0.9rem 0">')
        H('<div style="font-size:0.69rem;color:var(--text3);line-height:1.65">For research and educational purposes only. Always verify with a qualified agronomist before applying treatments.</div>')


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    left, right = st.columns([3, 2], gap="large")

    with left:
        selected_model = render_model_selector()
        ckpt = find_checkpoint(selected_model)
        model_ready = ckpt is not None
        model, device = None, torch.device("cpu")

        if model_ready:
            model, device = load_model(selected_model, str(ckpt))
            if model is None:
                model_ready = False

        meta = MODELS_META[selected_model]
        render_header(model_ready, meta["label"])
        render_kpis()
        render_scan_panel(model, device, model_ready, selected_model)
        render_class_reference()

    with right:
        render_sidebar(selected_model, ckpt)
        render_tips()


if __name__ == "__main__":
    main()