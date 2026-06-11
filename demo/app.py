"""
Streamlit web application for AgroVision project.
Provides an interactive interface for Cassava leaf disease prediction.
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from codes.model import create_model
from codes.utils import set_seed, get_device
from codes.config import SEED, NUM_CLASSES, INPUT_SIZE, MODEL_ARCHITECTURE


def load_model(model_path, num_classes=NUM_CLASSES, model_name=MODEL_ARCHITECTURE, device=None):
    """Load trained model from checkpoint."""
    if device is None:
        device = get_device()
    
    model = create_model(num_classes=num_classes, pretrained=False, model_name=model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, device


def predict_image(image, model, device, class_names):
    """Make prediction on a single image."""
    from torchvision import transforms
    
    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
    
    return predicted_class.item(), probabilities.cpu().numpy()[0]


def main():
    """Main Streamlit application."""
    st.set_page_config(page_title="AgroVision", page_icon="🌾")
    
    st.title("🌾 AgroVision - Cassava Leaf Disease Detection")
    st.markdown("**Identify Cassava leaf diseases using AI**")
    
    # Disease class names
    class_names = [
        "Cassava Brown Streak Disease",
        "Cassava Green Mottle Virus",
        "Cassava Mosaic Disease",
        "Cassava Leaf Blotch",
        "Healthy"
    ]
    
    # Set seed for reproducibility
    set_seed(SEED)
    device = get_device()
    
    # Sidebar
    st.sidebar.header("Configuration")
    model_path = st.sidebar.text_input(
        "Model checkpoint path",
        value="models/best_model.pth"
    )
    
    try:
        # Load model
        with st.spinner("Loading model..."):
            model, device = load_model(model_path, device=device)
        st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {str(e)}")
        st.stop()
    
    # Main content
    tab1, tab2 = st.tabs(["Upload Image", "About"])
    
    with tab1:
        st.subheader("Upload a Cassava leaf image")
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Make prediction
            with st.spinner("Making prediction..."):
                predicted_class, probabilities = predict_image(
                    image, model, device, class_names
                )
            
            # Display results
            with col2:
                st.subheader("Prediction Results")
                st.metric("Predicted Class", class_names[predicted_class])
                st.metric("Confidence", f"{probabilities[predicted_class]*100:.2f}%")
                
                # Probability distribution
                st.subheader("Class Probabilities")
                prob_dict = {name: prob for name, prob in zip(class_names, probabilities)}
                st.bar_chart(prob_dict)
    
    with tab2:
        st.subheader("About AgroVision")
        st.write("""
        **AgroVision** is an AI-powered system for detecting Cassava leaf diseases.
        
        ### Supported Diseases:
        1. **Cassava Brown Streak Disease** - Bacterial disease causing brown streaks
        2. **Cassava Green Mottle Virus** - Viral disease with green discoloration
        3. **Cassava Mosaic Disease** - Major viral disease with mosaic patterns
        4. **Cassava Leaf Blotch** - Fungal disease causing blotches
        5. **Healthy** - No disease detected
        
        ### How to Use:
        1. Go to the "Upload Image" tab
        2. Upload a Cassava leaf image (JPG, JPEG, or PNG)
        3. View the model's prediction and confidence score
        4. Check the probability distribution across all classes
        
        ### Model Details:
        - Architecture: EfficientNet-B0
        - Input Size: 224x224 pixels
        - Pre-trained on ImageNet
        
        For more information, visit the project repository.
        """)


if __name__ == "__main__":
    main()
