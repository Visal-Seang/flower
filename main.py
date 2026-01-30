"""
Flower Classification App
Using st.camera_input for reliable camera access on Streamlit Cloud
"""

import streamlit as st
import numpy as np
from PIL import Image
import tf_keras
from tf_keras.layers import DepthwiseConv2D
from typing import Dict, List, Tuple

# Configuration
st.set_page_config(
    page_title="Flower Classification",
    page_icon="🌸",
    layout="centered",
)

MODEL_PATH = "converted_keras/keras_model.h5"
LABELS_PATH = "converted_keras/labels.txt"
IMG_SIZE = (224, 224)


# Custom Keras Layer
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(**kwargs)


@st.cache_resource
def load_model():
    try:
        model = tf_keras.models.load_model(
            MODEL_PATH,
            custom_objects={"DepthwiseConv2D": CustomDepthwiseConv2D},
            compile=False,
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


@st.cache_data
def load_labels() -> Dict[int, str]:
    try:
        labels = {}
        with open(LABELS_PATH, "r") as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    labels[int(parts[0])] = parts[1]
        return labels
    except Exception as e:
        st.error(f"Error loading labels: {str(e)}")
        return {}


def preprocess_image(image: Image.Image) -> np.ndarray:
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(image, dtype=np.float32)
    arr = (arr / 127.5) - 1.0
    return np.expand_dims(arr, axis=0)


def predict_image(image: Image.Image, model, labels: Dict[int, str]):
    processed = preprocess_image(image)
    predictions = model.predict(processed, verbose=0)
    
    predicted_class = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class])
    label = labels.get(predicted_class, "Unknown")
    
    all_predictions = sorted(
        [(labels[i], float(predictions[0][i])) for i in labels],
        key=lambda x: -x[1],
    )
    
    return label, confidence, all_predictions


def main():
    # Title
    st.markdown("<h1 style='text-align: center;'>🌸 Flower Classification</h1>", unsafe_allow_html=True)

    # Load model and labels
    model = load_model()
    labels = load_labels()

    if model is None or not labels:
        st.error("Unable to load model or labels.")
        return

    st.success(f"Model loaded with {len(labels)} flower classes")

    # Tabs
    tab1, tab2 = st.tabs(["📷 Camera", "📁 Upload Image"])

    # TAB 1: CAMERA
    with tab1:
        st.markdown("### 📷 Take a Photo")
        st.markdown("Point your camera at a flower and click capture.")

        camera_image = st.camera_input("Take a picture of a flower")

        if camera_image is not None:
            image = Image.open(camera_image)

            with st.spinner("Analyzing..."):
                label, confidence, all_predictions = predict_image(image, model, labels)

            st.markdown("---")
            
            # Top prediction in blue
            st.markdown(
                f"<h3 style='text-align: center; color: #1E88E5;'>Top Prediction: {label} ({confidence:.2f})</h3>",
                unsafe_allow_html=True
            )

            # All predictions
            for lbl, prob in all_predictions:
                st.markdown(
                    f"<p style='text-align: center; margin: 5px 0;'>{lbl}: {prob:.2f}</p>",
                    unsafe_allow_html=True
                )

    # TAB 2: UPLOAD IMAGE
    with tab2:
        st.markdown("### 📁 Upload an Image")

        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            with st.spinner("Analyzing..."):
                label, confidence, all_predictions = predict_image(image, model, labels)

            st.markdown("---")
            
            # Top prediction in blue
            st.markdown(
                f"<h3 style='text-align: center; color: #1E88E5;'>Top Prediction: {label} ({confidence:.2f})</h3>",
                unsafe_allow_html=True
            )

            # All predictions
            for lbl, prob in all_predictions:
                st.markdown(
                    f"<p style='text-align: center; margin: 5px 0;'>{lbl}: {prob:.2f}</p>",
                    unsafe_allow_html=True
                )


if __name__ == "__main__":
    main()
