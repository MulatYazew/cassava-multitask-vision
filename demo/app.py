"""
AgroVision Africa — Streamlit Demo  (demo/app.py)
==================================================
Upload or capture a cassava leaf image to get a disease prediction with
class probabilities, Grad-CAM explanation, treatment advice, and
a per-class confidence breakdown.

Project layout expected:
  cassava-multitask-vision/
    codes/          ← config.py, model.py, data_handler.py, utils.py, gradcam.py
    demo/           ← this file
    models/
      best_model.pth  (or models/<run_name>/best_model.pth)

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

#  Path resolution 
# Support running from project root (demo/app.py) or from inside demo/
FILE = Path(__file__).resolve()
DEMO_DIR = FILE.parent
# Walk up until we find a directory that contains 'codes/'
ROOT = DEMO_DIR
for _candidate in [DEMO_DIR, DEMO_DIR.parent]:
    if (_candidate / "codes").exists():
        ROOT = _candidate
        break
sys.path.insert(0, str(ROOT))

#  Project imports
try:
    from codes.config import (  # noqa: E402
        NUM_CLASSES, INPUT_SIZE, MODELS_DIR, MODEL_ARCHITECTURE,
    )
    from codes.data_handler import CLASS_NAMES, CLASS_DESCRIPTIONS, get_transforms  # noqa: E402
    from codes.model import create_model  # noqa: E402
    from codes.utils import get_device  # noqa: E402
    _CODES_OK = True
except ModuleNotFoundError:
    _CODES_OK = False
    NUM_CLASSES = 5
    INPUT_SIZE = 224
    MODELS_DIR = ROOT / "models"
    MODEL_ARCHITECTURE = "efficientnet_v2_s"
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

#  Constants 
TREATMENTS: dict[int, str] = {
    0: "Remove and destroy infected plant material. Apply copper-based bactericide. Source disease-free cuttings for replanting.",
    1: "Rogue out infected plants immediately — no cure exists. Plant CBSD-tolerant varieties (e.g. Narocass 1). Control whitefly populations.",
    2: "Monitor closely; CGM is usually mild. Maintain healthy soil nutrition. Source certified virus-free planting material.",
    3: "Uproot severely infected plants. Plant CMD-resistant varieties (e.g. TMEB419, Serere). Eliminate whitefly vectors with neem-based spray.",
    4: "No action required. Continue regular crop scouting every 2 weeks. Maintain good soil health and weed management.",
}

SEVERITY: dict[int, str] = {0: "High", 1: "High", 2: "Medium", 3: "High", 4: "None"}

SEVERITY_COLOR: dict[str, str] = {
    "High":   "#ef4444",
    "Medium": "#f59e0b",
    "None":   "#22c55e",
}

CLASS_SHORT: dict[int, str] = {0: "CBB", 1: "CBSD", 2: "CGM", 3: "CMD", 4: "Healthy"}

#  Page config 
st.set_page_config(
    page_title="AgroVision Africa",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

#  Design tokens & global CSS 
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

/* ── base ── */
.stApp { background: var(--bg); font-family: var(--sans); color: var(--text); }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2.5rem 4rem; max-width: 1380px; }

/* ── typography ── */
h1,h2,h3 { font-family: var(--sans); color: var(--text); }

/* ── page header ── */
.av-header {
  display: flex; align-items: center; justify-content: space-between;
  padding-bottom: 1.25rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 2rem;
}
.av-wordmark {
  display: flex; align-items: center; gap: 14px;
}
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

/* ── KPI strip ── */
.kpi-strip {
  display: grid; grid-template-columns: repeat(4,1fr);
  border: 1px solid var(--border); border-radius: 12px;
  overflow: hidden; margin-bottom: 2rem;
}
.kpi {
  padding: 1rem 1.25rem;
  border-right: 1px solid var(--border);
  background: var(--surface);
}
.kpi:last-child { border-right: none; }
.kpi-v { font-family: var(--mono); font-size: 1.4rem; font-weight: 600; color: var(--text); }
.kpi-v.g { color: var(--green); }
.kpi-v.t { color: var(--teal); }
.kpi-v.a { color: var(--amber); }
.kpi-l { font-size: 0.7rem; color: var(--text3); margin-top: 4px; }

/* ── section label ── */
.sec-label {
  font-family: var(--mono); font-size: 0.72rem; font-weight: 600;
  letter-spacing: 0.12em; text-transform: uppercase; color: var(--text2);
  margin: 0 0 0.85rem;
  display: flex; align-items: center; gap: 8px;
}
.sec-dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--green); flex-shrink: 0;
}
.sec-num {
  background: var(--green-lo); color: var(--green);
  font-size: 0.6rem; width: 18px; height: 18px;
  display: flex; align-items: center; justify-content: center;
  border-radius: 3px; flex-shrink: 0;
}

/* ── upload zone ── */
.upload-zone {
  border: 1.5px dashed var(--border2);
  border-radius: 14px; padding: 1.75rem 1.5rem;
  text-align: center; background: var(--surface);
  margin-bottom: 0.5rem;
}
.upload-zone:hover { border-color: var(--green); }
.upload-icon { font-size: 2.2rem; margin-bottom: 0.5rem; }
.upload-title { font-size: 0.9rem; font-weight: 600; color: var(--text); margin-bottom: 0.2rem; }
.upload-hint  { font-size: 0.75rem; color: var(--text2); }

/* ── prediction card ── */
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

/* ── prob bars ── */
.bar-row { display: flex; align-items: center; gap: 9px; margin-bottom: 8px; }
.bar-lbl { font-size: 0.71rem; color: var(--text2); width: 48px; flex-shrink: 0; }
.bar-lbl.top { color: var(--text); font-weight: 600; }
.bar-track { flex: 1; height: 5px; background: var(--border); border-radius: 3px; overflow: hidden; }
.bar-fill  { height: 100%; border-radius: 3px; }
.bar-pct   { font-family: var(--mono); font-size: 0.67rem; color: var(--text3); width: 36px; text-align: right; }

/* ── treatment card ── */
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

/* ── severity ── */
.sev-badge {
  display: inline-flex; align-items: center; gap: 5px; margin-top: 5px;
}
.sev-dot { width: 7px; height: 7px; border-radius: 50%; }
.sev-txt { font-family: var(--mono); font-size: 0.63rem; }

/* ── tip card ── */
.tip {
  display: flex; align-items: flex-start; gap: 10px;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 10px 12px; margin-bottom: 7px;
}
.tip-icon { font-size: 1.05rem; flex-shrink: 0; margin-top: 1px; }
.tip-title { font-size: 0.77rem; font-weight: 600; color: var(--text); margin-bottom: 2px; }
.tip-body  { font-size: 0.71rem; color: var(--text2); line-height: 1.5; }

/* ── confidence legend ── */
.legend { display: flex; gap: 14px; flex-wrap: wrap; margin-top: 0.75rem; }
.leg-item { display: flex; align-items: center; gap: 5px; font-size: 0.69rem; color: var(--text3); }
.leg-dot  { width: 8px; height: 8px; border-radius: 2px; }

/* ── model card (sidebar) ── */
.mc-row { display: flex; justify-content: space-between; margin-bottom: 6px; }
.mc-k { font-size: 0.75rem; color: var(--text3); }
.mc-v { font-size: 0.75rem; color: var(--text); font-family: var(--mono); }

/* ── no-model banner ── */
.no-model {
  background: var(--surface); border: 1px solid var(--border2);
  border-left: 3px solid var(--amber);
  border-radius: 10px; padding: 1.25rem;
  font-size: 0.82rem; color: var(--text2); line-height: 1.7;
}

/* ── Streamlit widget overrides ── */
div[data-testid="stFileUploader"] > label { display: none !important; }
div[data-testid="stFileUploader"] section {
  background: transparent !important; border: none !important; padding: 0 !important;
}
div[data-testid="stFileUploader"] button {
  background: var(--green-lo) !important; border: 1px solid var(--green) !important;
  color: var(--green) !important; border-radius: 8px !important;
  width: 100% !important; margin-top: 10px !important;
  font-family: var(--sans) !important; font-size: 0.84rem !important;
}
div[data-testid="stFileUploader"] button:hover {
  background: var(--green) !important; color: #07100a !important;
}
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
div[data-testid="stCameraInput"] button {
  background: var(--green-lo) !important; border: 1px solid var(--green) !important;
  color: var(--green) !important; border-radius: 8px !important; width: 100% !important;
}
.stImage img { border-radius: 10px; }
div[data-testid="stExpander"] { background: var(--panel) !important; border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)


#  Helpers 
def H(html: str) -> None:
    """Render raw HTML."""
    st.markdown(html, unsafe_allow_html=True)


def conf_color(p: float) -> str:
    if p >= 0.75: return "#29e06b"
    if p >= 0.45: return "#f59e0b"
    return "#ef4444"


def conf_label(p: float) -> str:
    if p >= 0.75: return "High confidence"
    if p >= 0.45: return "Medium confidence"
    return "Low — verify manually"


#  Model loading 
@st.cache_resource(show_spinner="Loading model weights…")
def load_model(ckpt_path: str):
    """Load checkpoint; return (model, device) or (None, device) on failure."""
    device = get_device() if _CODES_OK else torch.device("cpu")
    try:
        model = create_model(num_classes=NUM_CLASSES, pretrained=False, model_name=MODEL_ARCHITECTURE)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        # Support checkpoints saved as raw state_dict or wrapped dict
        if isinstance(state, dict) and "net" in state:
            state = state["net"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        model.to(device).eval()
        return model, device
    except Exception as exc:
        st.error(f"Failed to load checkpoint: {exc}")
        return None, device


def run_inference(image: Image.Image, model, device) -> tuple[int, np.ndarray]:
    """Preprocess → forward → (pred_idx, probs array)."""
    img_np = np.array(image.convert("RGB"))
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


#  Checkpoint discovery 
def discover_checkpoints() -> list[str]:
    """Return list of available checkpoint paths relative to MODELS_DIR."""
    options = []
    models_path = Path(MODELS_DIR)
    if not models_path.exists():
        return options
    # Default flat checkpoint
    if (models_path / "best_model.pth").exists():
        options.append("(default)")
    # Sub-run checkpoints
    for sub in sorted(models_path.iterdir()):
        if sub.is_dir() and (sub / "best_model.pth").exists():
            options.append(sub.name)
    return options


def resolve_ckpt(choice: str) -> Path:
    p = Path(MODELS_DIR)
    return p / "best_model.pth" if choice == "(default)" else p / choice / "best_model.pth"


#  UI components 

def render_header(model_ready: bool):
    status = "MODEL READY" if model_ready else "NO CHECKPOINT"
    H(f"""
    <div class="av-header">
      <div class="av-wordmark">
        <div class="av-icon">🌿</div>
        <div>
          <div class="av-title">AgroVision Africa</div>
          <div class="av-sub">Cassava Leaf Disease Classifier · EfficientNet-V2-S</div>
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


def render_pred_card(pred_idx: int, probs: np.ndarray):
    top_p   = float(probs[pred_idx])
    col     = conf_color(top_p)
    sev     = SEVERITY[pred_idx]
    sev_col = SEVERITY_COLOR[sev]

    # probability bars, sorted by confidence
    sorted_idx = np.argsort(probs)[::-1]
    bars_html = ""
    for idx in sorted_idx:
        p     = float(probs[idx])
        w     = max(p * 100, 0.5)
        bc    = conf_color(p) if idx == pred_idx else "#1e3022"
        top   = " top" if idx == pred_idx else ""
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
      <div class="treat-title">⚕ Recommended Action</div>
      {TREATMENTS[pred_idx]}
    </div>
    <div class="legend">
      <span class="leg-item"><span class="leg-dot" style="background:#29e06b"></span>≥75% High</span>
      <span class="leg-item"><span class="leg-dot" style="background:#f59e0b"></span>45–74% Medium</span>
      <span class="leg-item"><span class="leg-dot" style="background:#ef4444"></span>&lt;45% Low — re-scan</span>
    </div>
    """)


def render_no_model():
    H(f"""
    <div class="no-model">
      <strong style="color:#f59e0b">No checkpoint found</strong><br><br>
      Expected location: <code>{MODELS_DIR}/best_model.pth</code><br>
      Run the training cells in <code>notebooks/AgroVision_Africa.ipynb</code> to generate a
      checkpoint, then restart this app.
    </div>
    """)


def render_scan_panel(model, device, model_ready: bool):
    H('<div class="sec-label"><span class="sec-num">2</span>Leaf Disease Scanner</div>')

    if not model_ready:
        render_no_model()
        return

    tab_up, tab_cam = st.tabs(["📁  Upload image", "📷  Take photo"])
    image = None

    with tab_up:
        H("""
        <div class="upload-zone">
          <div class="upload-icon">🍃</div>
          <div class="upload-title">Drop a cassava leaf photo here</div>
          <div class="upload-hint">JPG · PNG · any resolution · ideally well-lit and in-focus</div>
        </div>""")
        uploaded = st.file_uploader(
            "Choose file", type=["jpg", "jpeg", "png"],
            key="uploader", label_visibility="collapsed",
        )
        if uploaded:
            image = Image.open(uploaded).convert("RGB")

    with tab_cam:
        if "cam_on" not in st.session_state:
            st.session_state.cam_on = False
        H("""
        <div class="upload-zone" style="padding:1rem">
          <div class="upload-hint">Allow camera access, then tap <strong>Take photo</strong></div>
        </div>""")
        if st.button("📷  Open camera", key="btn_cam"):
            st.session_state.cam_on = True
        if st.session_state.cam_on:
            snap = st.camera_input("Camera", key="camera", label_visibility="collapsed")
            if st.button("✕  Close camera", key="btn_cam_off"):
                st.session_state.cam_on = False
            if snap:
                image = Image.open(snap).convert("RGB")

    if image:
        st.image(image, use_container_width=True, caption="Input image")
        with st.spinner("Running inference…"):
            pred_idx, probs = run_inference(image, model, device)
        render_pred_card(pred_idx, probs)
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
    H('<div class="sec-label" style="margin-top:1.5rem"><span class="sec-num">ℹ</span>Scanning Tips</div>')
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


def render_sidebar(checkpoint_options: list[str], selected: str | None):
    with st.sidebar:
        H('<div style="font-size:0.8rem;font-weight:700;color:#dff0e4;margin-bottom:0.75rem">📋 Model Card</div>')
        H(f"""
        <div>
          <div class="mc-row"><span class="mc-k">Architecture</span><span class="mc-v">{MODEL_ARCHITECTURE}</span></div>
          <div class="mc-row"><span class="mc-k">Classes</span><span class="mc-v">{NUM_CLASSES}</span></div>
          <div class="mc-row"><span class="mc-k">Input size</span><span class="mc-v">{INPUT_SIZE} × {INPUT_SIZE} px</span></div>
          <div class="mc-row"><span class="mc-k">Transfer</span><span class="mc-v">ImageNet → fine-tune</span></div>
          <div class="mc-row"><span class="mc-k">Dataset</span><span class="mc-v">Cassava Kaggle 2020</span></div>
        </div>
        <hr style="border-color:#1e3022;margin:0.9rem 0">""")

        H('<div style="font-size:0.8rem;font-weight:700;color:#dff0e4;margin-bottom:0.6rem">🌿 Classes</div>')
        for i in range(NUM_CLASSES):
            dot = SEVERITY_COLOR[SEVERITY[i]]
            H(f"""
            <div style="display:flex;align-items:center;gap:6px;margin-bottom:6px">
              <span style="width:7px;height:7px;border-radius:50%;background:{dot};display:inline-block;flex-shrink:0"></span>
              <span style="font-size:0.73rem"><b style="color:#dff0e4">{CLASS_SHORT[i]}</b>
              &nbsp;{CLASS_NAMES[i]}</span>
            </div>""")

        H('<hr style="border-color:#1e3022;margin:0.9rem 0">')
        H('<div style="font-size:0.8rem;font-weight:700;color:#dff0e4;margin-bottom:0.6rem">⚙ Checkpoint</div>')
        if checkpoint_options:
            st.selectbox(
                "Checkpoint", options=checkpoint_options,
                index=0, key="ckpt_choice", label_visibility="collapsed",
            )
        else:
            st.warning("No checkpoint found.")

        H('<hr style="border-color:#1e3022;margin:0.9rem 0">')
        H('<div style="font-size:0.69rem;color:#3d6348;line-height:1.6">This demo is for research and educational purposes. Always verify predictions with a qualified agronomist before applying treatments.</div>')


#  Main 
def main():
    checkpoints = discover_checkpoints()

    # Sidebar (renders checkpoint selector and sets st.session_state.ckpt_choice)
    render_sidebar(checkpoints, checkpoints[0] if checkpoints else None)

    # Resolve selected checkpoint path
    ckpt_choice  = st.session_state.get("ckpt_choice", checkpoints[0] if checkpoints else None)
    model_ready  = bool(ckpt_choice and checkpoints)
    model, device = (None, torch.device("cpu"))

    if model_ready:
        ckpt_path = str(resolve_ckpt(ckpt_choice))
        model, device = load_model(ckpt_path)
        if model is None:
            model_ready = False

    #  Header + KPIs 
    render_header(model_ready)
    render_kpis()

    #  Two-column layout 
    left, right = st.columns([3, 2], gap="large")

    with left:
        # Step 1 — crop selector
        H('<div class="sec-label"><span class="sec-num">1</span>Select Crop</div>')
        crop = st.selectbox(
            "Crop", ["Cassava 🌿", "Maize 🌽", "Tomato 🍅", "Sorghum 🌾", "Beans 🫘"],
            index=0, label_visibility="collapsed",
        )
        if "Cassava" not in crop:
            st.info(
                f"The model is trained on **cassava leaves only**. "
                f"Predictions for {crop.split()[0]} will still show cassava disease classes.",
            )

        # Step 2 — scan
        render_scan_panel(model, device, model_ready)

        # Step 3 — class reference
        render_class_reference()

    with right:
        render_tips()


if __name__ == "__main__":
    main()