import streamlit as st
import numpy as np
from PIL import Image
import tf_keras
from tf_keras.layers import DepthwiseConv2D
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import threading

st.set_page_config(page_title="Flower Classification", page_icon="🌸", layout="wide")

# WebRTC Configuration
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Global lock for thread-safe operations
prediction_lock = threading.Lock()


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


def predict_flower(image, model, labels):
    """Make prediction and return results"""
    processed = preprocess_image(image)
    predictions = model.predict(processed, verbose=0)

    predicted_class = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class])
    label = labels.get(predicted_class, "Unknown")

    all_results = sorted(
        [(labels[i], float(predictions[0][i])) for i in labels],
        key=lambda x: -x[1],
    )

    return label, confidence, all_results


def display_results(label, confidence, all_results):
    """Display prediction results with proper formatting"""
    emoji = "🌸" if confidence > 0.7 else "🤔" if confidence > 0.4 else "❓"

    if confidence > 0.7:
        st.success(f"## {emoji} {label}")
    elif confidence > 0.4:
        st.warning(f"## {emoji} {label}")
    else:
        st.error(f"## {emoji} {label}")

    st.metric("Confidence", f"{confidence * 100:.1f}%")

    st.markdown("**All Predictions:**")
    for lbl, prob in all_results:
        icon = "🟢" if prob > 0.7 else "🟠" if prob > 0.4 else "🔴"
        st.progress(prob, text=f"{icon} {lbl}: {prob * 100:.1f}%")


class VideoProcessor:
    """Video frame processor for real-time flower detection"""

    def __init__(self):
        self.model = None
        self.labels = None
        self.frame_count = 0
        self.prediction_result = None

    def set_model(self, model, labels):
        """Set the model and labels for prediction"""
        self.model = model
        self.labels = labels

    def recv(self, frame):
        """Process each video frame"""
        img = frame.to_ndarray(format="bgr24")

        # Process every 30th frame to reduce computation
        self.frame_count += 1
        if self.frame_count % 30 == 0 and self.model is not None:
            try:
                # Convert BGR to RGB
                img_rgb = img[:, :, ::-1]
                pil_img = Image.fromarray(img_rgb)

                # Make prediction
                processed = preprocess_image(pil_img)
                predictions = self.model.predict(processed, verbose=0)

                predicted_class = np.argmax(predictions[0])
                confidence = float(predictions[0][predicted_class])
                label = self.labels.get(predicted_class, "Unknown")

                all_results = sorted(
                    [(self.labels[i], float(predictions[0][i])) for i in self.labels],
                    key=lambda x: -x[1],
                )

                # Store prediction in thread-safe manner
                with prediction_lock:
                    self.prediction_result = (label, confidence, all_results)

            except Exception as e:
                print(f"Error in frame processing: {e}")

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def main():
    st.title("🌸 Flower Classification")
    st.caption("Take a photo or upload an image of a Tulip, Rose, or Sunflower!")

    # Load model and labels
    try:
        model = load_model()
        labels = load_labels()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # Sidebar
    with st.sidebar:
        st.header("📖 Instructions")
        st.markdown(
            """
        **Real-Time Mode:**
        1. Click **START** button
        2. Allow camera access
        3. Point at a flower
        4. See live predictions!
        
        **Snapshot Mode:**
        1. Take a photo or upload
        2. Get instant results
        """
        )

        st.markdown("---")
        st.header("🎨 Confidence Legend")
        st.markdown(
            """
        - 🟢 **Green** = High (>70%)
        - 🟠 **Orange** = Medium (40-70%)
        - 🔴 **Red** = Low (<40%)
        """
        )

        st.markdown("---")
        st.header("🌷 Supported Flowers")
        st.markdown(
            """
        - 🌷 Tulip
        - 🌹 Rose
        - 🌻 Sunflower
        """
        )

    # Main content with tabs
    tab1, tab2, tab3 = st.tabs(
        ["🎥 Real-Time Detection", "📸 Camera Capture", "📁 Upload Image"]
    )

    with tab1:
        st.markdown("### 🎥 Real-Time Flower Detection")
        st.info("🎥 Live video feed with automatic flower detection!")

        col1, col2 = st.columns([3, 2])

        with col1:
            # Initialize video processor
            if "video_processor" not in st.session_state:
                st.session_state.video_processor = VideoProcessor()
                st.session_state.video_processor.set_model(model, labels)

            # WebRTC streamer for real-time video
            webrtc_ctx = webrtc_streamer(
                key="flower-detection",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_processor_factory=lambda: st.session_state.video_processor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

            st.markdown(
                "**Status:** "
                + ("🟢 Active" if webrtc_ctx.state.playing else "🔴 Stopped")
            )

        with col2:
            st.markdown("### 📊 Live Results")

            # Display predictions
            if webrtc_ctx.state.playing:
                result_placeholder = st.empty()

                # Check for new predictions
                if st.session_state.video_processor.prediction_result:
                    with prediction_lock:
                        label, confidence, all_results = (
                            st.session_state.video_processor.prediction_result
                        )

                    with result_placeholder.container():
                        display_results(label, confidence, all_results)
                else:
                    result_placeholder.info("👀 Waiting for detection...")
            else:
                st.info("👆 Click 'START' above to begin real-time detection!")

        st.markdown(
            """
        ---
        **Tips for best results:**
        - Allow camera access when prompted
        - Ensure good lighting
        - Center the flower in frame
        - Hold camera steady
        - Get close to the flower
        - Detection updates every ~1 second
        """
        )

    with tab2:
        st.markdown("### 📸 Take a Photo")
        st.info("Click the button below to take a photo of a flower!")

        # Camera input
        camera_photo = st.camera_input("Take a picture")

        if camera_photo:
            # Read the image
            image = Image.open(camera_photo)

            col1, col2 = st.columns([1, 1])

            with col1:
                st.image(image, caption="Captured Image", use_container_width=True)

            with col2:
                with st.spinner("Analyzing..."):
                    label, confidence, all_results = predict_flower(
                        image, model, labels
                    )

                display_results(label, confidence, all_results)

        st.markdown(
            """
        ---
        **Tips for best results:**
        - Ensure good lighting
        - Center the flower in frame
        - Hold camera steady
        - Get close to the flower
        """
        )

    with tab3:
        st.markdown("### 📁 Upload an Image")

        uploaded = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded:
            image = Image.open(uploaded)

            col1, col2 = st.columns([1, 1])

            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)

            with col2:
                with st.spinner("Analyzing..."):
                    label, confidence, all_results = predict_flower(
                        image, model, labels
                    )

                display_results(label, confidence, all_results)


if __name__ == "__main__":
    main()
