import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import tf_keras
from tf_keras.layers import DepthwiseConv2D
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
import av
import threading

st.set_page_config(page_title="Flower Classification", page_icon="🌸", layout="wide")


# Custom layer for model compatibility
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(**kwargs)


@st.cache_resource
def load_model():
    return tf_keras.models.load_model(
        "converted_keras/keras_model.h5",
        custom_objects={"DepthwiseConv2D": CustomDepthwiseConv2D},
        compile=False,
    )


@st.cache_data
def load_labels():
    labels = {}
    with open("converted_keras/labels.txt", "r") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                labels[int(parts[0])] = parts[1]
    return labels


def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((224, 224), Image.LANCZOS)
    arr = np.array(image, dtype=np.float32)
    arr = (arr / 127.5) - 1
    return np.expand_dims(arr, axis=0)


# Load model and labels globally
MODEL = None
LABELS = None


# Thread-safe result storage
class ResultStore:
    def __init__(self):
        self.lock = threading.Lock()
        self.label = ""
        self.confidence = 0.0
        self.all_results = []
        self.updated = False

    def update(self, label, confidence, all_results):
        with self.lock:
            self.label = label
            self.confidence = confidence
            self.all_results = all_results
            self.updated = True

    def get(self):
        with self.lock:
            return {
                "label": self.label,
                "confidence": self.confidence,
                "all_results": self.all_results.copy(),
                "updated": self.updated,
            }

    def mark_read(self):
        with self.lock:
            self.updated = False


# Global result store
RESULT_STORE = ResultStore()


class FlowerDetector(VideoProcessorBase):
    def __init__(self):
        self.model = MODEL
        self.labels = LABELS

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert frame to numpy array (BGR format)
        img = frame.to_ndarray(format="bgr24")

        # Convert BGR to RGB for prediction
        img_rgb = img[:, :, ::-1].copy()
        pil_image = Image.fromarray(img_rgb)

        # Run prediction
        try:
            processed = preprocess_image(pil_image)
            predictions = self.model.predict(processed, verbose=0)

            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            label = self.labels.get(predicted_class, "Unknown")

            # Store results globally
            all_results = sorted(
                [(self.labels[i], float(predictions[0][i])) for i in self.labels],
                key=lambda x: -x[1],
            )

            RESULT_STORE.update(label, confidence, all_results)

            # Choose color based on confidence (BGR format for OpenCV)
            if confidence > 0.7:
                color = (0, 255, 0)  # Green
            elif confidence > 0.4:
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 0, 255)  # Red

            # Draw result on frame using PIL
            pil_frame = Image.fromarray(img)
            draw = ImageDraw.Draw(pil_frame)

            # Draw background rectangle for text
            draw.rectangle([10, 10, 400, 70], fill=color)

            # Draw text
            text = f"{label}: {confidence * 100:.1f}%"
            draw.text((20, 20), text, fill=(255, 255, 255))

            # Draw border around frame
            border_width = 8
            w, h = pil_frame.size
            draw.rectangle([0, 0, w - 1, border_width], fill=color)
            draw.rectangle([0, h - border_width, w - 1, h - 1], fill=color)
            draw.rectangle([0, 0, border_width, h - 1], fill=color)
            draw.rectangle([w - border_width, 0, w - 1, h - 1], fill=color)

            img = np.array(pil_frame)

        except Exception as e:
            pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def main():
    global MODEL, LABELS

    # Simple centered title
    st.markdown(
        "<h1 style='text-align: center;'>🌸 Flower Classification</h1>",
        unsafe_allow_html=True,
    )

    # Initialize session state
    if "detection_count" not in st.session_state:
        st.session_state.detection_count = 0

    # Load model and labels
    try:
        MODEL = load_model()
        LABELS = load_labels()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # Main content with tabs
    tab1, tab2 = st.tabs(["📹 Real-Time Detection", "📁 Upload Image"])

    with tab1:
        # Create columns to center camera
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            # WebRTC streamer for real-time video
            ctx = webrtc_streamer(
                key="flower-detection",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=FlowerDetector,
                media_stream_constraints={
                    "video": {
                        "width": {"ideal": 640},
                        "height": {"ideal": 480},
                        "frameRate": {"ideal": 30},
                        "facingMode": "environment",
                    },
                    "audio": False,
                },
                async_processing=True,
                rtc_configuration={
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                },
            )

            # Get latest results from store
            results = RESULT_STORE.get()

            # Display results below camera (centered)
            if ctx.state.playing and results["label"]:
                label = results["label"]
                conf = results["confidence"]
                all_results = results["all_results"]

                # Top Prediction in blue
                st.markdown(
                    f"<h3 style='text-align: center; color: #1E88E5;'>Top Prediction: {label} ({conf:.2f})</h3>",
                    unsafe_allow_html=True,
                )

                # All predictions as simple centered text
                for lbl, prob in all_results:
                    st.markdown(
                        f"<p style='text-align: center; margin: 5px 0;'>{lbl}: {prob:.2f}</p>",
                        unsafe_allow_html=True,
                    )

                # Update counter to trigger re-render
                st.session_state.detection_count += 1

            elif ctx.state.playing:
                st.markdown(
                    "<p style='text-align: center; color: gray;'>🔍 Point camera at a flower to detect...</p>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<p style='text-align: center; color: gray;'>▶️ Click START to begin detection</p>",
                    unsafe_allow_html=True,
                )

    with tab2:
        st.markdown(
            "<h3 style='text-align: center;'>📁 Upload an Image</h3>",
            unsafe_allow_html=True,
        )

        # Center the file uploader
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            uploaded = st.file_uploader(
                "Choose an image...", type=["jpg", "jpeg", "png"]
            )

            if uploaded:
                image = Image.open(uploaded)

                # Display image centered
                st.image(image, caption="Uploaded Image", use_container_width=True)

                with st.spinner("Analyzing..."):
                    processed = preprocess_image(image)
                    predictions = MODEL.predict(processed, verbose=0)

                    predicted_class = np.argmax(predictions[0])
                    confidence = float(predictions[0][predicted_class])
                    label = LABELS.get(predicted_class, "Unknown")

                    results = sorted(
                        [(LABELS[i], float(predictions[0][i])) for i in LABELS],
                        key=lambda x: -x[1],
                    )

                # Top Prediction in blue
                st.markdown(
                    f"<h3 style='text-align: center; color: #1E88E5;'>Top Prediction: {label} ({confidence:.2f})</h3>",
                    unsafe_allow_html=True,
                )

                # All predictions as simple centered text
                for lbl, prob in results:
                    st.markdown(
                        f"<p style='text-align: center; margin: 5px 0;'>{lbl}: {prob:.2f}</p>",
                        unsafe_allow_html=True,
                    )


if __name__ == "__main__":
    main()
