import streamlit as st
import numpy as np
from PIL import Image
import tf_keras
from tf_keras.layers import DepthwiseConv2D


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
        "Upload an image to identify if it's a **Tulip**, **Rose**, or **Sunflower**"
    )

    # Load model and labels
    try:
        model = load_model()
        labels = load_labels()
    except Exception as e:
        st.error(f"Error loading model or labels: {e}")
        return

    # File uploader - only accepts single image
    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"], accept_multiple_files=False
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
