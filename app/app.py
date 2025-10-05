import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import cv2
import keras

# ------------------------------
# Configuration
# ------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CLASS_MODEL_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "models", "resnet50_brain_tumor.h5"))
SEG_MODEL_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "models", "best_unet_model.keras"))

IMG_SIZE = (224, 224)
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
SEGMENTATION_THRESHOLD = 0.1

# --- DEFINE OR IMPORT YOUR CUSTOM OBJECTS HERE ---

# You can define them directly, or import them if they are in another file.
# Option 1: Define them directly (ensure this matches model_segmentation.py)
@keras.saving.register_keras_serializable()
def dice_coeff(y_true, y_pred, smooth=100):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    return dice

@keras.saving.register_keras_serializable()
class FocalLoss(keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        bce_exp = tf.keras.backend.exp(-bce)
        focal_loss_val = self.alpha * tf.keras.backend.pow((1 - bce_exp), self.gamma) * bce
        return focal_loss_val

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "gamma": self.gamma, "alpha": self.alpha}

# Option 2 (Preferred): Import them to avoid duplication (if you structure your project as a package)
# from model_segmentation import dice_coeff, FocalLoss

# ------------------------------
# Load models (cached for Streamlit)
# ------------------------------
@st.cache_resource
def load_model(path):
    """
    Loads a Keras model, explicitly providing the custom objects it needs.
    """
    try:
        # The key is to provide a dictionary of all custom objects used in the model
        custom_objects = {
            "dice_coeff": dice_coeff,
            "FocalLoss": FocalLoss,
            # Add any other custom objects here (layers, metrics, etc.)
        }
        model = tf.keras.models.load_model(path, custom_objects=custom_objects)
        print(f"Model loaded successfully from {path}")
        return model
    except Exception as e:
        st.error(f"Error loading model from {path}: {e}")
        return None

# Load models
class_model = load_model(CLASS_MODEL_PATH)
seg_model = load_model(SEG_MODEL_PATH)

# ------------------------------
# Image preprocessing & Prediction (Keep the rest of your code the same)
# ------------------------------
def preprocess_image(image: Image.Image, target_size=IMG_SIZE):
    image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def get_segmentation_mask(model, image, threshold):
    img_array = preprocess_image(image)
    prob_map = model.predict(img_array, verbose=0)[0]
    mask = (prob_map > threshold).astype(np.uint8)
    return mask

def create_segmentation_visualization(original_image, mask):
    original_np = np.array(original_image.convert("RGB"))
    mask = mask.squeeze()
    mask_resized = cv2.resize(mask, (original_np.shape[1], original_np.shape[0]), interpolation=cv2.INTER_NEAREST)

    colored_mask = np.zeros_like(original_np)
    colored_mask[mask_resized == 1] = [255, 0, 0] # Red for tumor

    overlay = cv2.addWeighted(original_np, 0.7, colored_mask, 0.3, 0)
    return overlay, mask_resized

# ------------------------------
# Streamlit UI (Keep the rest of your code the same)
# ------------------------------
st.set_page_config(page_title="Brain Tumor Analysis", page_icon="🧠", layout="wide")

st.sidebar.title("🧠 Brain Tumor Analyzer")
st.sidebar.markdown("Upload an MRI scan to classify the tumor type and segment the tumor region.")

st.title("Brain Tumor Classification & Segmentation")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded MRI")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Analysis Results")

        # Classification
        if class_model is not None:
            with st.spinner("Classifying tumor type..."):
                input_tensor = preprocess_image(image)
                preds = class_model.predict(input_tensor, verbose=0)
                pred_idx = np.argmax(preds)
                pred_class = CLASS_NAMES[pred_idx]
                confidence = np.max(preds)

            st.write("##### Classification")
            if pred_class == "No Tumor":
                st.success(f"**Prediction:** {pred_class} ({confidence:.2%})")
            else:
                st.error(f"**Prediction:** {pred_class} ({confidence:.2%})")
        else:
            st.warning("Classification model not loaded.")

        st.write("---")

        # Segmentation
        if seg_model is not None:
            st.write("##### Segmentation")
            threshold = st.slider("Segmentation Threshold", 0.01, 0.9, SEGMENTATION_THRESHOLD, 0.01)

            with st.spinner("Segmenting tumor..."):
                mask = get_segmentation_mask(seg_model, image, threshold)
                overlay, final_mask = create_segmentation_visualization(image, mask)

            st.image(overlay, caption=f"Tumor Overlay (Threshold: {threshold})", use_container_width=True)

            # Tumor Area Calculation
            tumor_pixels = np.sum(final_mask)
            total_pixels = final_mask.shape[0] * final_mask.shape[1]
            if total_pixels > 0:
                tumor_percentage = (tumor_pixels / total_pixels) * 100
                st.metric("Tumor Area", f"{tumor_percentage:.2f}%")

        else:
            st.warning("Segmentation model not loaded.")
else:
    st.info("Please upload an image to begin analysis.")