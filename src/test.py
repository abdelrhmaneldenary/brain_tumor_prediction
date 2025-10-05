import tensorflow as tf
import os

print(f"Using TensorFlow version: {tf.__version__}")

# --- Copy all your custom functions here ---
def dice_coeff(y_true, y_pred, smooth=100):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    return dice

def dice_loss(y_true, y_pred):
    return 1 - dice_coeff(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice

def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fn(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        bce_exp = tf.keras.backend.exp(-bce)
        focal_loss_val = alpha * tf.keras.backend.pow((1 - bce_exp), gamma) * bce
        return focal_loss_val
    return focal_loss_fn

# --- Define the path to your model ---
# Make sure this path is correct relative to where you run the script
SEG_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "best_unet_segmentation.h5"))

# --- This is our best guess for the custom objects ---
CUSTOM_OBJECTS = {
    'dice_coeff': dice_coeff,
    'bce_dice_loss': bce_dice_loss,
    'focal_loss': focal_loss()
}

# --- Attempt to load the model ---
try:
    print(f"Attempting to load model from: {SEG_MODEL_PATH}")
    model = tf.keras.models.load_model(SEG_MODEL_PATH, custom_objects=CUSTOM_OBJECTS)
    print("\n✅✅✅  MODEL LOADED SUCCESSFULLY! ✅✅✅")
    model.summary()
except Exception as e:
    print(f"\n❌❌❌ LOADING FAILED. Error: {e} ❌❌❌")