import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ------------------------------
# Configuration
# ------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models/resnet50_brain_tumor.h5")  # or "model.h5"
IMG_SIZE = (224, 224)  # must match training

# Class names (update these if different in your dataset)
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# ------------------------------
# Load model once
# ------------------------------
@st.cache_resource
def load_model():
    try:
        if MODEL_PATH.endswith(".h5"):
            model = tf.keras.models.load_model(MODEL_PATH)
        else:
            model = tf.keras.models.load_model(MODEL_PATH)  # SavedModel format
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# ------------------------------
# Preprocessing
# ------------------------------
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)  # shape (1, H, W, C)
    img_array = img_array / 255.0  # normalize (must match training preprocessing)
    return img_array

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🧠 Brain Tumor Classification")
st.write("Upload an MRI scan to predict the tumor type.")

uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI", use_container_width=True)

    if model is not None:
        # Preprocess & predict
        input_tensor = preprocess_image(image)
        preds = model.predict(input_tensor)
        predicted_class = CLASS_NAMES[np.argmax(preds)]
        confidence = np.max(preds)

        # Show results
        st.subheader("Prediction Results")
        st.write(f"**Predicted Class:** {predicted_class}")
        st.write(f"**Confidence:** {confidence:.2f}")

        # Probability bar chart
        st.bar_chart({CLASS_NAMES[i]: float(preds[0][i]) for i in range(len(CLASS_NAMES))})
    else:
        st.error("Model not loaded. Check the MODEL_PATH.")
