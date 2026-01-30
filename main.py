"""
Flower Classification App - Real-Time Detection
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import tf_keras
from tf_keras.layers import DepthwiseConv2D
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
import av
import threading
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


# Global storage for results
class ResultStore:
    def __init__(self):
        self.lock = threading.Lock()
        self.label = ""
        self.confidence = 0.0
        self.all_results = []

    def update(self, label, confidence, all_results):
        with self.lock:
            self.label = label
            self.confidence = confidence
            self.all_results = all_results

    def get(self):
        with self.lock:
            return self.label, self.confidence, self.all_results.copy()


RESULT_STORE = ResultStore()


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


# Global model and labels for video processor
MODEL = None
LABELS = None


class FlowerDetector(VideoProcessorBase):
    def __init__(self):
        self.model = MODEL
        self.labels = LABELS
        self.frame_count = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        self.frame_count += 1
        if self.frame_count % 3 != 0:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        try:
            img_rgb = img[:, :, ::-1].copy()
            pil_image = Image.fromarray(img_rgb)

            processed = preprocess_image(pil_image)
            predictions = self.model.predict(processed, verbose=0)

            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            label = self.labels.get(predicted_class, "Unknown")

            all_results = sorted(
                [(self.labels[i], float(predictions[0][i])) for i in self.labels],
                key=lambda x: -x[1],
            )

            RESULT_STORE.update(label, confidence, all_results)

            # Draw on frame
            pil_frame = Image.fromarray(img)
            draw = ImageDraw.Draw(pil_frame)

            if confidence > 0.7:
                color = (0, 255, 0)
            elif confidence > 0.4:
                color = (0, 165, 255)
            else:
                color = (0, 0, 255)

            text = f"{label}: {confidence * 100:.1f}%"
            draw.rectangle([10, 10, 300, 45], fill=color)
            draw.text((20, 15), text, fill=(255, 255, 255))

            w, h = pil_frame.size
            for i in range(5):
                draw.rectangle([i, i, w - 1 - i, h - 1 - i], outline=color)

            img = np.array(pil_frame)

        except Exception:
            pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def main():
    global MODEL, LABELS
    # Title
    st.markdown(
        "<h1 style='text-align: center;'>🌸 Flower Classification</h1>",
        unsafe_allow_html=True,
    )

    # Load model and labels
    MODEL = load_model()
    LABELS = load_labels()

    if MODEL is None or not LABELS:
        st.error("Unable to load model or labels.")
        return

    st.success(f"Model loaded with {len(LABELS)} flower classes")

    # Tabs
    tab1, tab2 = st.tabs(["📹 Real-Time", "📁 Upload Image"])

    # TAB 1: REAL-TIME
    with tab1:
        st.markdown("### 📹 Real-Time Detection")
        st.markdown("Point your camera at a flower for live detection.")

        # WebRTC streamer
        ctx = webrtc_streamer(
            key="flower-detection",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=FlowerDetector,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
            rtc_configuration={
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun1.l.google.com:19302"]},
                ]
            },
        )

        # Display results below video
        if ctx.state.playing:
            label, confidence, all_results = RESULT_STORE.get()

            if label:
                st.markdown(
                    f"<h3 style='text-align: center; color: #1E88E5;'>Top Prediction: {label} ({confidence:.2f})</h3>",
                    unsafe_allow_html=True,
                )
                for lbl, prob in all_results:
                    st.markdown(
                        f"<p style='text-align: center;'>{lbl}: {prob:.2f}</p>",
                        unsafe_allow_html=True,
                    )
            else:
                st.info("🔍 Point camera at a flower...")
        else:
            st.info("▶️ Click START to begin real-time detection")

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
                label, confidence, all_predictions = predict_image(image, MODEL, LABELS)

            st.markdown("---")

            # Top prediction in blue
            st.markdown(
                f"<h3 style='text-align: center; color: #1E88E5;'>Top Prediction: {label} ({confidence:.2f})</h3>",
                unsafe_allow_html=True,
            )

            # All predictions
            for lbl, prob in all_predictions:
                st.markdown(
                    f"<p style='text-align: center; margin: 5px 0;'>{lbl}: {prob:.2f}</p>",
                    unsafe_allow_html=True,
                )


if __name__ == "__main__":
    main()
