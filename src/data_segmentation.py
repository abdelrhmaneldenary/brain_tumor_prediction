import tensorflow as tf
from tensorflow.keras import layers
from pathlib import Path
import kagglehub
import os
import tensorflow_io as tfio  # pip install tensorflow-io

AUTOTUNE = tf.data.AUTOTUNE


def load_segmentation_dataset(batch_size=8, image_size=(224, 224), seed=123, fallback_dir=None):
    """
    Load LGG MRI segmentation dataset (robust TIFF handling).
    Returns: (train_ds, val_ds)
    """

    # Try download via kagglehub
    try:
        dataset_path = Path(kagglehub.dataset_download("mateuszbuda/lgg-mri-segmentation"))
        data_root = dataset_path / "lgg-mri-segmentation" / "kaggle_3m"
        print(f"✅ Dataset found at: {data_root}")
    except Exception as e:
        print("⚠️ Download failed:", str(e))
        if fallback_dir is None:
            raise RuntimeError("No fallback directory provided and download failed.")
        data_root = Path(fallback_dir)
        print(f"Using fallback directory: {data_root}")

    # Collect .tif files recursively
    all_images = sorted([p for p in data_root.rglob("*.tif") if "_mask" not in p.stem])
    all_masks = sorted([p for p in data_root.rglob("*.tif") if "_mask" in p.stem])

    if not all_images or not all_masks:
        raise FileNotFoundError(f"No images or masks found in {data_root}")

    print(f"🧩 Found {len(all_images)} images and {len(all_masks)} masks")

    # Create mask lookup and pair
    mask_dict = {m.stem.replace("_mask", ""): m for m in all_masks}
    paired = [(img, mask_dict[img.stem]) for img in all_images if img.stem in mask_dict]

    if not paired:
        raise ValueError("No image-mask pairs found. Check naming consistency in dataset.")

    print(f"✅ Paired {len(paired)} image-mask samples")

    img_paths = [str(p[0]) for p in paired]
    msk_paths = [str(p[1]) for p in paired]
    dataset = tf.data.Dataset.from_tensor_slices((img_paths, msk_paths))

    # Helper functions: ensure RGB and single-channel mask
    def _ensure_rgb(img):
        """Return an image with shape [...,3] and dtype float32 in [0,1]."""
        img = tf.image.convert_image_dtype(img, tf.float32)  # handles uint8/uint16 -> [0,1]
        ch = tf.shape(img)[-1]

        # If single channel -> convert to RGB. Else take first 3 channels (handles RGBA).
        return tf.cond(
            tf.equal(ch, 1),
            lambda: tf.image.grayscale_to_rgb(img),
            lambda: img[..., :3]
        )

    def _ensure_mask_single(mask):
        """Return a single-channel mask (float32 0/1) regardless of input channels."""
        mask = tf.image.convert_image_dtype(mask, tf.float32)  # scale to [0,1]
        ch = tf.shape(mask)[-1]

        # If single-channel keep it; if multi-channel take first channel
        return tf.cond(
            tf.equal(ch, 1),
            lambda: mask,
            lambda: tf.expand_dims(mask[..., 0], axis=-1)
        )

    def process_path(img_path, msk_path):
        # Read and decode TIFF (returns dtype like uint16 or uint8)
        img = tf.io.read_file(img_path)
        img = tfio.experimental.image.decode_tiff(img)  # shape: [H,W,C]
        img = _ensure_rgb(img)
        img = tf.image.resize(img, image_size)
        # img is float32 in [0,1] already

        mask = tf.io.read_file(msk_path)
        mask = tfio.experimental.image.decode_tiff(mask)
        mask = _ensure_mask_single(mask)
        # Use nearest neighbor for masks to avoid interpolating labels
        mask = tf.image.resize(mask, image_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        mask = tf.cast(mask > 0.5, tf.float32)  # binary mask: 0.0 or 1.0

        return img, mask

    dataset = dataset.map(process_path, num_parallel_calls=AUTOTUNE)

    # Split into validation and train (80/20)
    total = len(paired)
    val_count = int(0.2 * total)
    val_ds = dataset.take(val_count)
    train_ds = dataset.skip(val_count)

    # Augmentation
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ])

    def augment(img, mask):
        return data_augmentation(img), mask

    train_ds = train_ds.map(augment, num_parallel_calls=AUTOTUNE)

    # Shuffle, batch, prefetch
    train_ds = train_ds.shuffle(1000, seed=seed).batch(batch_size).prefetch(AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(AUTOTUNE)

    print("✅ Dataset successfully loaded and preprocessed.")
    return train_ds, val_ds


if __name__ == "__main__":
    train_ds, val_ds = load_segmentation_dataset()
    print("✅ Dataset ready for training.")