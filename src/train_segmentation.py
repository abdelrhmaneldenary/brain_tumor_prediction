from data_segmentation import load_segmentation_dataset
# Import the class-based FocalLoss and other necessities
from model_segmentation import unet_model, dice_coeff, FocalLoss
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
import os
import tensorflow as tf

def train_segmentation(epochs=50, use_focal_loss=False):
    train_ds, val_ds = load_segmentation_dataset(
        batch_size=8,
        image_size=(224, 224),
        fallback_dir="/home/denary/.cache/kagglehub/datasets/mateuszbuda/lgg-mri-segmentation/versions/2/lgg-mri-segmentation/kaggle_3m"
    )

    model = unet_model(use_focal_loss=use_focal_loss)

    checkpoint_path = os.path.join("models", "best_unet_model.keras")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor="val_dice_coeff", mode="max", save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor="val_dice_coeff", mode="max", factor=0.5, patience=5, min_lr=1e-7, verbose=1)
    early_stop = EarlyStopping(monitor="val_dice_coeff", mode="max", patience=15, restore_best_weights=True, verbose=1)
    csv_logger = CSVLogger(os.path.join("models", "segmentation_log.csv"))

    print("\n--- Starting training ---")
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[checkpoint, reduce_lr, early_stop, csv_logger])
    
    print(f"\n✅ Training complete. Best model saved to {checkpoint_path}")
    return model, history

if __name__ == "__main__":
    # Now you can train with Focal Loss without future loading errors
    model, history = train_segmentation(epochs=50, use_focal_loss=True)