"""
AgroVision Africa — Streamlit Demo

Upload or capture cassava leaf images to get disease predictions with
class probabilities, Grad-CAM explanations, treatment advice, and
per-class confidence breakdowns.

Run:  streamlit run demo/app.py   (from project root)
      OR
      streamlit run app.py        (from inside demo/)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

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

SEVERITY: dict[int, str]       = {0: "High", 1: "High", 2: "Medium", 3: "High", 4: "None"}
SEVERITY_COLOR: dict[str, str] = {"High": "#ef4444", "Medium": "#f59e0b", "None": "#22c55e"}
CLASS_SHORT: dict[int, str]    = {0: "CBB", 1: "CBSD", 2: "CGM", 3: "CMD", 4: "Healthy"}

MODELS_META: dict[str, dict] = {
    "resnet50": {
        "label": "ResNet-50",
        "desc":  "Classical CNN baseline — fast and well-understood.",
        "tag":   "CNN · baseline",
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

OOD_MAX_PROB = 0.35   # flag as out-of-distribution if max softmax prob below this
OOD_ENTROPY  = 0.92   # flag as out-of-distribution if normalized entropy above this

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
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
  --bg:       #07100a;
  --surface:  #0e1a11;
  --panel:    #162019;
  --border:   #1e3022;
  --border2:  #2a4230;
  --green:    #29e06b;
  --green-lo: #29e06b18;
  --teal:     #00cfb4;
  --amber:    #f59e0b;
  --red:      #ef4444;
  --text:     #dff0e4;
  --text2:    #7fa88a;
  --text3:    #3d6348;
  --mono:     'IBM Plex Mono', monospace;
  --sans:     'Inter', sans-serif;
}

/* base */
.stApp { background: var(--bg); font-family: var(--sans); color: var(--text); }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2.5rem 4rem; max-width: 1380px; }
h1,h2,h3 { font-family: var(--sans); color: var(--text); }

/* page header */
.av-header {
  display: flex; align-items: center; justify-content: space-between;
  padding-bottom: 1.25rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 2rem;
}
.av-wordmark { display: flex; align-items: center; gap: 14px; }
.av-icon {
  width: 44px; height: 44px; border-radius: 10px;
  background: var(--green); display: flex; align-items: center;
  justify-content: center; font-size: 22px; flex-shrink: 0;
}
.av-title { font-size: 1.15rem; font-weight: 700; color: var(--text); margin: 0; line-height: 1.2; }
.av-sub   { font-size: 0.68rem; color: var(--text3); letter-spacing: 0.1em; text-transform: uppercase; margin: 2px 0 0; }
.av-pill  {
  background: var(--green-lo); border: 1px solid #29e06b30;
  color: var(--green); font-family: var(--mono); font-size: 0.65rem;
  padding: 5px 12px; border-radius: 4px; letter-spacing: 0.06em;
}

/* KPI strip */
.kpi-strip {
  display: grid; grid-template-columns: repeat(4,1fr);
  border: 1px solid var(--border); border-radius: 12px;
  overflow: hidden; margin-bottom: 2rem;
}
.kpi { padding: 1rem 1.25rem; border-right: 1px solid var(--border); background: var(--surface); }
.kpi:last-child { border-right: none; }
.kpi-v { font-family: var(--mono); font-size: 1.4rem; font-weight: 600; color: var(--text); }
.kpi-v.g { color: var(--green); }
.kpi-v.t { color: var(--teal); }
.kpi-v.a { color: var(--amber); }
.kpi-l { font-size: 0.7rem; color: var(--text3); margin-top: 4px; }

/* section label */
.sec-label {
  font-family: var(--mono); font-size: 0.72rem; font-weight: 600;
  letter-spacing: 0.12em; text-transform: uppercase; color: var(--text2);
  margin: 0 0 0.85rem;
  display: flex; align-items: center; gap: 8px;
}
.sec-num {
  background: var(--green-lo); color: var(--green);
  font-size: 0.6rem; width: 18px; height: 18px;
  display: flex; align-items: center; justify-content: center;
  border-radius: 3px; flex-shrink: 0;
}

/* prediction card */
.pred-card {
  background: var(--panel);
  border: 1px solid var(--border2);
  border-radius: 14px; padding: 1.35rem;
  margin-top: 1rem;
}
.pred-eyebrow {
  font-family: var(--mono); font-size: 0.6rem;
  letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--text3); margin-bottom: 4px;
}
.pred-name { font-size: 1.15rem; font-weight: 700; color: var(--text); }
.conf-pill {
  padding: 4px 13px; border-radius: 20px;
  font-family: var(--mono); font-size: 0.75rem; font-weight: 600;
  white-space: nowrap;
}
.pred-desc {
  font-size: 0.8rem; color: var(--text2); line-height: 1.65;
  margin: 0.9rem 0 1.1rem;
}
.divider { border: none; border-top: 1px solid var(--border); margin: 0.85rem 0; }

/* prob bars */
.bar-row { display: flex; align-items: center; gap: 9px; margin-bottom: 8px; }
.bar-lbl { font-size: 0.71rem; color: var(--text2); width: 48px; flex-shrink: 0; }
.bar-lbl.top { color: var(--text); font-weight: 600; }
.bar-track { flex: 1; height: 5px; background: var(--border); border-radius: 3px; overflow: hidden; }
.bar-fill  { height: 100%; border-radius: 3px; }
.bar-pct   { font-family: var(--mono); font-size: 0.67rem; color: var(--text3); width: 36px; text-align: right; }

/* treatment card */
.treatment {
  background: var(--surface);
  border: 1px solid var(--border2);
  border-left: 3px solid var(--green);
  border-radius: 10px; padding: 0.9rem 1rem;
  margin-top: 0.8rem;
  font-size: 0.8rem; color: var(--text2); line-height: 1.7;
}
.treat-title {
  font-family: var(--mono); font-size: 0.6rem;
  letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--green); margin-bottom: 5px;
}

/* severity */
.sev-badge { display: inline-flex; align-items: center; gap: 5px; margin-top: 5px; }
.sev-dot { width: 7px; height: 7px; border-radius: 50%; }
.sev-txt { font-family: var(--mono); font-size: 0.63rem; }

/* tip card */
.tip {
  display: flex; align-items: flex-start; gap: 10px;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 10px 12px; margin-bottom: 7px;
}
.tip-icon { font-size: 1.05rem; flex-shrink: 0; margin-top: 1px; }
.tip-title { font-size: 0.77rem; font-weight: 600; color: var(--text); margin-bottom: 2px; }
.tip-body  { font-size: 0.71rem; color: var(--text2); line-height: 1.5; }

/* confidence legend */
.legend { display: flex; gap: 14px; flex-wrap: wrap; margin-top: 0.75rem; }
.leg-item { display: flex; align-items: center; gap: 5px; font-size: 0.69rem; color: var(--text3); }
.leg-dot  { width: 8px; height: 8px; border-radius: 2px; }

/* model card (sidebar) */
.mc-row { display: flex; justify-content: space-between; margin-bottom: 6px; }
.mc-k { font-size: 0.75rem; color: var(--text3); }
.mc-v { font-size: 0.75rem; color: var(--text); font-family: var(--mono); }

/* no-model banner */
.no-model {
  background: var(--surface); border: 1px solid var(--border2);
  border-left: 3px solid var(--amber);
  border-radius: 10px; padding: 1.25rem;
  font-size: 0.82rem; color: var(--text2); line-height: 1.7;
}

/* OOD banner */
.ood-banner {
  background: #1a0f00;
  border: 1px solid #f59e0b40;
  border-left: 3px solid var(--amber);
  border-radius: 10px; padding: 1rem 1.1rem;
  display: flex; align-items: flex-start; gap: 12px;
  margin-top: 1rem;
}
.ood-icon  { font-size: 1.5rem; flex-shrink: 0; line-height: 1; }
.ood-title { font-weight: 700; color: var(--amber); margin-bottom: 4px; font-size: 0.9rem; }
.ood-body  { font-size: 0.78rem; color: var(--text2); line-height: 1.6; }

/* batch result mini-card */
.batch-card {
  background: var(--panel);
  border: 1px solid var(--border2);
  border-radius: 10px; padding: 0.65rem;
  margin-top: 0.4rem; text-align: center;
}
.batch-label { font-size: 0.78rem; font-weight: 600; color: var(--text); margin-bottom: 2px; }
.batch-conf  { font-family: var(--mono); font-size: 0.7rem; }

/* model selector info card */
.model-info-card {
  background: var(--surface);
  border: 1px solid var(--border2);
  border-left: 3px solid var(--green);
  border-radius: 10px; padding: 0.75rem 1rem;
  margin-bottom: 1.25rem;
}
.model-tag  {
  font-family: var(--mono); font-size: 0.6rem; color: var(--green);
  letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 4px;
}
.model-desc { font-size: 0.82rem; color: var(--text2); }

/* native Streamlit file uploader — style the dropzone directly */
div[data-testid="stFileUploaderDropzone"] {
  border: 1.5px dashed var(--border2) !important;
  border-radius: 14px !important;
  padding: 1.75rem 1.5rem !important;
  background: var(--surface) !important;
  cursor: pointer !important;
  transition: border-color 0.15s ease, background 0.15s ease !important;
}
div[data-testid="stFileUploaderDropzone"]:hover {
  border-color: var(--green) !important;
  background: var(--green-lo) !important;
}
div[data-testid="stFileUploader"] > label { display: none !important; }
div[data-testid="stFileUploader"] section {
  background: transparent !important; border: none !important; padding: 0 !important;
}

/* camera input visual boundary */
div[data-testid="stCameraInput"] {
  border: 1.5px dashed var(--border2);
  border-radius: 14px;
  padding: 0.75rem;
  background: var(--surface);
}
div[data-testid="stCameraInput"] video { border-radius: 10px !important; }
div[data-testid="stCameraInput"] button {
  background: var(--green-lo) !important; border: 1px solid var(--green) !important;
  color: var(--green) !important; border-radius: 8px !important; width: 100% !important;
  margin-top: 8px !important;
}

/* misc overrides */
.stButton > button {
  background: var(--surface) !important; border: 1px solid var(--border2) !important;
  color: var(--text2) !important; border-radius: 8px !important;
  font-family: var(--sans) !important; font-size: 0.84rem !important;
  width: 100%;
}
.stButton > button:hover {
  border-color: var(--green) !important; color: var(--green) !important;
  background: var(--green-lo) !important;
}
.stTabs [data-baseweb="tab-list"] {
  background: transparent !important; border-bottom: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important; color: var(--text3) !important;
  font-family: var(--sans) !important; font-size: 0.84rem !important;
  border-bottom: 2px solid transparent !important; border-radius: 0 !important;
}
.stTabs [aria-selected="true"] {
  color: var(--green) !important; border-bottom-color: var(--green) !important;
  background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1rem !important; }
div[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }
div[data-testid="stSidebar"] * { color: var(--text2) !important; }
div[data-testid="stSelectbox"] > div > div {
  background: var(--surface) !important; border-color: var(--border2) !important; color: var(--text) !important;
}
div[data-testid="stRadio"] > div { gap: 6px !important; }
div[data-testid="stRadio"] label { color: var(--text2) !important; font-size: 0.9rem !important; }
.stImage img { border-radius: 10px; }
div[data-testid="stExpander"] { background: var(--panel) !important; border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────────────────────────
def H(html: str) -> None:
    st.markdown(html, unsafe_allow_html=True)


def conf_color(p: float) -> str:
    if p >= 0.75: return "#29e06b"
    if p >= 0.45: return "#f59e0b"
    return "#ef4444"


def conf_label(p: float) -> str:
    if p >= 0.75: return "High confidence"
    if p >= 0.45: return "Medium confidence"
    return "Low — verify manually"


# ── checkpoint helpers ────────────────────────────────────────────────────────
def find_checkpoint(model_name: str) -> Path | None:
    """Return first existing checkpoint for model_name, or None."""
    base = Path(MODELS_DIR)
    per_model = base / model_name / "best_model.pth"
    if per_model.exists():
        return per_model
    default = base / "best_model.pth"
    if default.exists():
        return default
    return None


# ── model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model weights…")
def load_model(model_name: str, ckpt_path: str):
    """Load checkpoint; return (model, device) or (None, device) on failure."""
    device = get_device() if _CODES_OK else torch.device("cpu")
    if not _CODES_OK:
        return None, device
    try:
        model = build_model(model_name, num_classes=NUM_CLASSES, dropout=0.3)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        if isinstance(state, dict) and "net" in state:
            state = state["net"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        model.to(device).eval()
        return model, device
    except Exception:
        return None, device


# ── inference ─────────────────────────────────────────────────────────────────
def run_inference(image: Image.Image, model, device) -> tuple[int, np.ndarray]:
    """Return (pred_idx, probs array) for one PIL image."""
    img_np = np.array(image.convert("RGB"))
    try:
        if _CODES_OK:
            transform = get_transforms(INPUT_SIZE, augment=False)
            tensor = transform(image=img_np)["image"].unsqueeze(0).to(device)
        else:
            from torchvision import transforms as T
            transform = T.Compose([
                T.Resize((INPUT_SIZE, INPUT_SIZE)),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            tensor = transform(Image.fromarray(img_np)).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = F.softmax(model(tensor), dim=1).cpu().numpy().flatten()
        return int(np.argmax(probs)), probs
    except Exception:
        uniform = np.ones(NUM_CLASSES, dtype=np.float32) / NUM_CLASSES
        return 0, uniform


def check_ood(probs: np.ndarray) -> bool:
    """Return True if the image is likely out-of-distribution (not a cassava leaf)."""
    max_prob = float(np.max(probs))
    n = len(probs)
    entropy = -float(np.sum(probs * np.log(probs + 1e-12))) / np.log(n)
    return max_prob < OOD_MAX_PROB or entropy > OOD_ENTROPY


def run_gradcam(
    image: Image.Image,
    model,
    device,
    pred_idx: int,
    model_name: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (overlay_rgb, resized_rgb) or (None, None) on failure."""
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
    H(f"""
    <div class="av-header">
      <div class="av-wordmark">
        <div class="av-icon">🌿</div>
        <div>
          <div class="av-title">AgroVision Africa</div>
          <div class="av-sub">Cassava Leaf Disease Classifier · {model_label}</div>
        </div>
      </div>
      <span class="av-pill">{status}</span>
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
    """Render model radio selector; return chosen model key."""
    H('<div class="sec-label"><span class="sec-num">1</span>Select Model</div>')

    options = list(MODELS_META.keys())
    labels  = [MODELS_META[k]["label"] for k in options]

    chosen_label = st.radio(
        "Model",
        labels,
        index=1,  # default: EfficientNet-V2-S
        key="model_radio",
        label_visibility="collapsed",
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
      <strong style="color:#f59e0b">No checkpoint found for {MODELS_META[model_name]['label']}</strong><br><br>
      Expected location: <code>{ckpt_hint}</code><br>
      Train the model in <code>notebooks/AgroVision_Africa.ipynb</code>, then restart this app.
    </div>
    """)


def render_ood_banner():
    H("""
    <div class="ood-banner">
      <div class="ood-icon">⚠</div>
      <div>
        <div class="ood-title">Not a cassava leaf</div>
        <div class="ood-body">
          The model cannot confidently classify this image as cassava.
          It may be a different crop, an object, or a poor-quality photo.<br>
          Please upload a clear, well-lit photo of a cassava leaf.
        </div>
      </div>
    </div>
    """)


def render_pred_card(pred_idx: int, probs: np.ndarray):
    top_p   = float(probs[pred_idx])
    col     = conf_color(top_p)
    sev     = SEVERITY[pred_idx]
    sev_col = SEVERITY_COLOR[sev]

    sorted_idx = np.argsort(probs)[::-1]
    bars_html = ""
    for idx in sorted_idx:
        p   = float(probs[idx])
        w   = max(p * 100, 0.5)
        bc  = conf_color(p) if idx == pred_idx else "#1e3022"
        top = " top" if idx == pred_idx else ""
        bars_html += f"""
        <div class="bar-row">
          <span class="bar-lbl{top}">{CLASS_SHORT[idx]}</span>
          <div class="bar-track"><div class="bar-fill" style="width:{w:.1f}%;background:{bc}"></div></div>
          <span class="bar-pct">{p*100:.1f}%</span>
        </div>"""

    H(f"""
    <div class="pred-card">
      <div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:0.6rem">
        <div>
          <div class="pred-eyebrow">Predicted Condition</div>
          <div class="pred-name">{CLASS_NAMES[pred_idx]}</div>
          <div class="sev-badge">
            <span class="sev-dot" style="background:{sev_col}"></span>
            <span class="sev-txt" style="color:{sev_col}">Severity: {sev}</span>
          </div>
        </div>
        <span class="conf-pill" style="background:{col}18;color:{col};border:1px solid {col}40">
          {top_p*100:.1f}% &nbsp;·&nbsp; {conf_label(top_p)}
        </span>
      </div>
      <div class="pred-desc">{CLASS_DESCRIPTIONS[pred_idx]}</div>
      <hr class="divider">
      <div class="pred-eyebrow" style="margin-bottom:10px">Class Probabilities</div>
      {bars_html}
    </div>
    <div class="treatment">
      <div class="treat-title">Recommended Action</div>
      {TREATMENTS[pred_idx]}
    </div>
    <div class="legend">
      <span class="leg-item"><span class="leg-dot" style="background:#29e06b"></span>≥75% High</span>
      <span class="leg-item"><span class="leg-dot" style="background:#f59e0b"></span>45–74% Medium</span>
      <span class="leg-item"><span class="leg-dot" style="background:#ef4444"></span>&lt;45% Low — re-scan</span>
    </div>
    """)


def render_gradcam_tab(image: Image.Image, model, device, pred_idx: int, model_name: str):
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
      border-radius:10px;font-size:0.79rem;color:var(--text2);line-height:1.7">
      <span style="font-family:var(--mono);font-size:0.6rem;letter-spacing:0.1em;
        text-transform:uppercase;color:var(--teal)">What am I looking at?</span><br>
      <strong style="color:var(--text)">Grad-CAM</strong> highlights the leaf regions that most
      influenced the prediction.
      <span style="color:#ef4444">Red/yellow</span> areas drove the classification;
      <span style="color:#3b82f6">blue</span> areas had little impact.
    </div>
    <div style="margin-top:0.55rem;font-family:var(--mono);font-size:0.65rem;
      color:var(--text3);letter-spacing:0.05em">
      Explaining → <strong style="color:var(--text2)">{CLASS_NAMES[pred_idx]}</strong>
    </div>
    """)


def render_batch_results(images: list[Image.Image], model, device, model_name: str):
    """Show a grid of results for a batch of images."""
    H('<div class="sec-label" style="margin-top:1rem">Batch Results</div>')
    n_cols = min(len(images), 3)
    cols = st.columns(n_cols)
    for i, img in enumerate(images):
        with cols[i % n_cols]:
            st.image(img, use_container_width=True)
            pred_idx, probs = run_inference(img, model, device)
            is_ood = check_ood(probs)
            if is_ood:
                H("""
                <div class="batch-card" style="border-left:3px solid var(--amber)">
                  <div class="batch-label" style="color:var(--amber)">Not cassava</div>
                  <div class="batch-conf" style="color:var(--text3)">OOD detected</div>
                </div>
                """)
            else:
                top_p = float(probs[pred_idx])
                col   = conf_color(top_p)
                H(f"""
                <div class="batch-card">
                  <div class="batch-label">{CLASS_SHORT[pred_idx]}</div>
                  <div class="batch-conf" style="color:{col}">{top_p*100:.1f}%</div>
                </div>
                """)


def render_scan_panel(model, device, model_ready: bool, model_name: str):
    H('<div class="sec-label"><span class="sec-num">2</span>Leaf Disease Scanner</div>')

    if not model_ready:
        render_no_model(model_name)
        return

    tab_up, tab_cam = st.tabs(["📁  Upload image(s)", "📷  Take photo"])
    images: list[Image.Image] = []

    with tab_up:
        uploaded = st.file_uploader(
            "Drop cassava leaf photos here",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="uploader",
            label_visibility="collapsed",
        )
        if uploaded:
            images = [Image.open(f).convert("RGB") for f in uploaded]

    with tab_cam:
        snap = st.camera_input(
            "Point at a cassava leaf and take a photo",
            key="camera",
            label_visibility="collapsed",
        )
        if snap:
            images = [Image.open(snap).convert("RGB")]

    if not images:
        return

    if len(images) == 1:
        image = images[0]
        st.image(image, use_container_width=True, caption="Input image")
        with st.spinner("Running inference…"):
            pred_idx, probs = run_inference(image, model, device)
            is_ood = check_ood(probs)

        if is_ood:
            render_ood_banner()
        else:
            tab_results, tab_gradcam = st.tabs(["📊  Results", "🔥  Grad-CAM Explanation"])
            with tab_results:
                render_pred_card(pred_idx, probs)
            with tab_gradcam:
                render_gradcam_tab(image, model, device, pred_idx, model_name)

    else:
        with st.spinner(f"Analysing {len(images)} images…"):
            render_batch_results(images, model, device, model_name)

    if st.button("↻  Scan another image"):
        st.rerun()


def render_class_reference():
    H('<div class="sec-label" style="margin-top:1.75rem"><span class="sec-num">3</span>Disease Class Reference</div>')
    for i in range(NUM_CLASSES):
        sev_col = SEVERITY_COLOR[SEVERITY[i]]
        with st.expander(f"{CLASS_SHORT[i]} — {CLASS_NAMES[i]}"):
            H(f"""
            <div style="color:var(--text2,#7fa88a);font-size:0.82rem;line-height:1.65;margin-bottom:0.75rem">
              {CLASS_DESCRIPTIONS[i]}
            </div>
            <div class="treat-title">Recommended action</div>
            <div style="color:var(--text2,#7fa88a);font-size:0.82rem;line-height:1.65">
              {TREATMENTS[i]}
            </div>
            <div class="sev-badge" style="margin-top:8px">
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
        H('<div style="font-size:0.8rem;font-weight:700;color:#dff0e4;margin-bottom:0.75rem">Model Card</div>')
        ckpt_str = str(ckpt) if ckpt else "— not found —"
        meta = MODELS_META[selected_model]
        H(f"""
        <div>
          <div class="mc-row"><span class="mc-k">Architecture</span><span class="mc-v">{meta['label']}</span></div>
          <div class="mc-row"><span class="mc-k">Classes</span><span class="mc-v">{NUM_CLASSES}</span></div>
          <div class="mc-row"><span class="mc-k">Input size</span><span class="mc-v">{INPUT_SIZE} × {INPUT_SIZE} px</span></div>
          <div class="mc-row"><span class="mc-k">Transfer</span><span class="mc-v">ImageNet → fine-tune</span></div>
          <div class="mc-row"><span class="mc-k">Dataset</span><span class="mc-v">Cassava Kaggle 2020</span></div>
          <div class="mc-row"><span class="mc-k">Checkpoint</span><span class="mc-v" style="word-break:break-all;font-size:0.65rem">{Path(ckpt_str).name if ckpt else 'not found'}</span></div>
        </div>
        <hr style="border-color:#1e3022;margin:0.9rem 0">""")

        H('<div style="font-size:0.8rem;font-weight:700;color:#dff0e4;margin-bottom:0.6rem">Classes</div>')
        for i in range(NUM_CLASSES):
            dot = SEVERITY_COLOR[SEVERITY[i]]
            H(f"""
            <div style="display:flex;align-items:center;gap:6px;margin-bottom:6px">
              <span style="width:7px;height:7px;border-radius:50%;background:{dot};display:inline-block;flex-shrink:0"></span>
              <span style="font-size:0.73rem"><b style="color:#dff0e4">{CLASS_SHORT[i]}</b>
              &nbsp;{CLASS_NAMES[i]}</span>
            </div>""")

        H('<hr style="border-color:#1e3022;margin:0.9rem 0">')
        H('<div style="font-size:0.69rem;color:#3d6348;line-height:1.6">This demo is for research and educational purposes. Always verify predictions with a qualified agronomist before applying treatments.</div>')


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    # ── two-column layout ──
    left, right = st.columns([3, 2], gap="large")

    with left:
        # Step 1 — model selection
        selected_model = render_model_selector()

        # resolve checkpoint & load model
        ckpt = find_checkpoint(selected_model)
        model_ready = ckpt is not None
        model, device = None, torch.device("cpu")

        if model_ready:
            model, device = load_model(selected_model, str(ckpt))
            if model is None:
                model_ready = False

        # header & KPIs (after model selection so we know the label)
        meta = MODELS_META[selected_model]
        render_header(model_ready, meta["label"])
        render_kpis()

        # Step 2 — scanner
        render_scan_panel(model, device, model_ready, selected_model)

        # Step 3 — class reference
        render_class_reference()

    with right:
        render_sidebar(selected_model, ckpt)
        render_tips()


if __name__ == "__main__":
    main()
