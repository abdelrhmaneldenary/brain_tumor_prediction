import tensorflow as tf
import kagglehub
from tensorflow.keras import layers
from pathlib import Path
import os

AUTOTUNE = tf.data.AUTOTUNE

def load_dataset(batch_size=32, image_size=(224, 224), seed=123):
    # Download dataset
    dataset_path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
    print(dataset_path)
    train_dir = dataset_path / "Training"
    val_dir   = dataset_path / "Testing"   # your dataset's validation folder

    # Training dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        label_mode='categorical',
        batch_size=batch_size,
        image_size=image_size,
        seed=seed
    )

    # Validation dataset
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        label_mode='categorical',
        batch_size=batch_size,
        image_size=image_size,
        seed=seed
    )

    # Stronger augmentation to reduce overfitting
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15),          # ±30% rotation
        layers.RandomTranslation(0.15, 0.15),# up to 15% shift
        layers.RandomZoom(0.2),              # up to 20% zoom
        layers.RandomContrast(0.2),          # stronger contrast
        layers.RandomBrightness(0.2),        # adjust brightness
    ], name="data_augmentation")

    # Apply augmentation only to training
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=AUTOTUNE
    )

    # Prefetch for performance
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds   = val_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds


if __name__ == "__main__":
    train_ds, val_ds = load_dataset()
    print("✅ Dataset loaded with augmentation (Training + Validation)")
