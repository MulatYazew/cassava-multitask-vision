"""
AgroVision Africa — Streamlit Demo
Upload a cassava leaf image and get a disease prediction with
class probabilities, using the best checkpoint trained in
AgroVision_Africa.ipynb.

Run with:  streamlit run app.py   (from the project root)
"""

import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

#  Project imports (code reuse — no redefinitions) 
ROOT = Path(__file__).resolve().parent.parent  # cassava-multitask-vision/ (demo/ is one level down)
sys.path.insert(0, str(ROOT))

from codes.config import NUM_CLASSES, INPUT_SIZE, MODELS_DIR, MODEL_ARCHITECTURE  # noqa: E402
from codes.data_handler import CLASS_NAMES, CLASS_DESCRIPTIONS, get_transforms      # noqa: E402
from codes.model import create_model                                               # noqa: E402
from codes.utils import get_device                                                # noqa: E402


#  Page config 
st.set_page_config(page_title="AgroVision Africa — Cassava Diagnosis", page_icon="🌿", layout="centered")
st.title("🌿 AgroVision Africa")
st.subheader("Cassava Leaf Disease Classifier")
st.write(
    "Upload a photo of a cassava leaf and the model will predict the "
    "most likely disease class, along with confidence scores for all classes."
)


#  Sidebar — model selection 
st.sidebar.header("Model")
available_models = sorted(
    [p.name for p in Path(MODELS_DIR).glob("*") if (p / "best_model.pth").exists()]
) if Path(MODELS_DIR).exists() else []

if Path(MODELS_DIR / "best_model.pth").exists():
    available_models = ["(default)"] + available_models

model_choice = st.sidebar.selectbox(
    "Checkpoint",
    options=available_models or ["No checkpoint found"],
    help="Trained checkpoints saved by train.py / the notebook.",
)


#  Cache model loading 
@st.cache_resource(show_spinner="Loading model...")
def load_model(checkpoint_name: str):
    device = get_device()
    model = create_model(num_classes=NUM_CLASSES, pretrained=False, model_name=MODEL_ARCHITECTURE)

    if checkpoint_name == "(default)":
        ckpt_path = Path(MODELS_DIR) / "best_model.pth"
    else:
        ckpt_path = Path(MODELS_DIR) / checkpoint_name / "best_model.pth"

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device


#  Main demo 
uploaded_file = st.file_uploader("Upload a cassava leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and available_models and available_models[0] != "No checkpoint found":
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    model, device = load_model(model_choice)

    # Same preprocessing pipeline used at validation/inference time.
    transform = get_transforms(INPUT_SIZE, augment=False)
    tensor = transform(image=np.array(image))["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy().flatten()

    pred_idx = int(np.argmax(probs))

    st.markdown("### Prediction")
    st.success(f"**{CLASS_NAMES[pred_idx]}** ({probs[pred_idx] * 100:.1f}% confidence)")
    st.caption(CLASS_DESCRIPTIONS[pred_idx])

    st.markdown("### Class probabilities")
    prob_table = {
        CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)
    }
    sorted_probs = dict(sorted(prob_table.items(), key=lambda kv: kv[1], reverse=True))
    st.bar_chart(sorted_probs)

    with st.expander("Raw probabilities"):
        for name, p in sorted_probs.items():
            st.write(f"{name}: {p:.4f}")

elif not available_models or available_models[0] == "No checkpoint found":
    st.warning(
        "No trained checkpoint found in `models/`. "
        "Run the training cells in `AgroVision_Africa.ipynb` first, "
        "then restart this app."
    )
else:
    st.info("Upload an image to get a prediction.")


st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Classes:**\n" + "\n".join(f"- {v}" for v in CLASS_NAMES.values())
)