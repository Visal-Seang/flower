import streamlit as st
import numpy as np
from PIL import Image
import tf_keras
from tf_keras.layers import DepthwiseConv2D
import cv2
import time


# Custom DepthwiseConv2D to handle 'groups' parameter issue
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        # Remove 'groups' parameter if present
        kwargs.pop("groups", None)
        super().__init__(**kwargs)


# Load the model and labels
@st.cache_resource
def load_model():
    model_path = "converted_keras/keras_model.h5"
    # Load with custom objects to handle compatibility issues
    model = tf_keras.models.load_model(
        model_path,
        custom_objects={"DepthwiseConv2D": CustomDepthwiseConv2D},
        compile=False,
    )
    return model


@st.cache_data
def load_labels():
    labels_path = "converted_keras/labels.txt"
    labels = {}
    with open(labels_path, "r") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                labels[int(parts[0])] = parts[1]
    return labels


def preprocess_image(image):
    # Convert to RGB if needed (in case of RGBA or grayscale)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize image to 224x224 (standard for Teachable Machine models)
    image = image.resize((224, 224), Image.LANCZOS)

    # Convert to array
    img_array = np.array(image, dtype=np.float32)

    # Normalize to [-1, 1] range (Teachable Machine uses this range)
    img_array = (img_array / 127.5) - 1

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def predict_flower(model, image, labels):
    # Preprocess the image
    processed_img = preprocess_image(image)

    # Make prediction
    predictions = model.predict(processed_img)

    # Get the predicted class
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    # Get all predictions with confidence scores
    results = []
    for i, prob in enumerate(predictions[0]):
        if i in labels:
            results.append((labels[i], prob))

    results.sort(key=lambda x: x[1], reverse=True)

    return labels[predicted_class], confidence, results


# Streamlit UI
def main():
    st.set_page_config(
        page_title="Flower Classification", page_icon="🌸", layout="centered"
    )

    st.title("🌸 Flower Classification App")
    st.write(
        "Upload an image or use webcam to identify if it's a **Tulip**, **Rose**, or **Sunflower**"
    )

    # Load model and labels
    try:
        model = load_model()
        labels = load_labels()
    except Exception as e:
        st.error(f"Error loading model or labels: {e}")
        return

    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📷 Webcam", "📁 Upload Image"])

    with tab1:
        st.write("### Real-time Flower Detection")

        # Webcam controls
        col1, col2 = st.columns(2)
        with col1:
            start_webcam = st.button("🎥 Start Webcam", use_container_width=True)
        with col2:
            stop_webcam = st.button("⏹️ Stop Webcam", use_container_width=True)

        # Initialize session state for webcam
        if "webcam_running" not in st.session_state:
            st.session_state.webcam_running = False

        if start_webcam:
            st.session_state.webcam_running = True
        if stop_webcam:
            st.session_state.webcam_running = False

        # Placeholders for webcam feed and results
        frame_placeholder = st.empty()
        result_placeholder = st.empty()
        progress_placeholder = st.empty()

        if st.session_state.webcam_running:
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                st.error(
                    "❌ Could not open webcam. Please check your camera connection."
                )
            else:
                st.info("📸 Webcam is running... Press 'Stop Webcam' to end.")

                while st.session_state.webcam_running:
                    ret, frame = cap.read()

                    if not ret:
                        st.error("Failed to capture frame from webcam")
                        break

                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Convert to PIL Image for prediction
                    pil_image = Image.fromarray(frame_rgb)

                    # Make prediction
                    predicted_label, confidence, all_results = predict_flower(
                        model, pil_image, labels
                    )

                    # Draw prediction on frame
                    cv2.putText(
                        frame_rgb,
                        f"{predicted_label}: {confidence * 100:.1f}%",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 255, 0),
                        3,
                    )

                    # Add colored border based on confidence
                    border_color = (
                        (0, 255, 0)
                        if confidence > 0.7
                        else (255, 165, 0) if confidence > 0.4 else (255, 0, 0)
                    )
                    frame_rgb = cv2.copyMakeBorder(
                        frame_rgb, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=border_color
                    )

                    # Display frame
                    frame_placeholder.image(
                        frame_rgb, channels="RGB", use_container_width=True
                    )

                    # Display results
                    with result_placeholder.container():
                        st.success(
                            f"**Detected: {predicted_label}** (Confidence: {confidence * 100:.1f}%)"
                        )

                    # Display progress bars for all predictions
                    with progress_placeholder.container():
                        for label, prob in all_results:
                            st.progress(float(prob), text=f"{label}: {prob * 100:.1f}%")

                    # Small delay to reduce CPU usage
                    time.sleep(0.1)

                cap.release()

    with tab2:
        # File uploader - only accepts single image
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
        )

        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)

            col1, col2 = st.columns(2)

            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)

            with col2:
                # Make prediction
                with st.spinner("Analyzing..."):
                    predicted_label, confidence, all_results = predict_flower(
                        model, image, labels
                    )

                # Display results
                st.success(f"**Prediction: {predicted_label}**")
                st.metric("Confidence", f"{confidence * 100:.2f}%")

                # Show all predictions
                st.write("**All Predictions:**")
                for label, prob in all_results:
                    st.progress(float(prob), text=f"{label}: {prob * 100:.2f}%")


if __name__ == "__main__":
    main()
