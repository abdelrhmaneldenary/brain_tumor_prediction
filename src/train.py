from data import load_dataset
from model import build_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
import os

def train(epochs=20,trainable_layers=6):
    # Load datasets
    train_ds, val_ds = load_dataset(batch_size=32)

    # Build model
    model = build_model(trainable_layers=trainable_layers)

    # --- Callbacks ---
    # earlystop = EarlyStopping(
    #     monitor="val_loss",
    #     patience=3,
    #     restore_best_weights=True,
    #     verbose=1
    # )

    checkpoint_path = os.path.join("..", "models", "best_resnet50.h5")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )

    log_file_path = os.path.join( "models", "training_log.csv")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    csv_logger = CSVLogger(log_file_path, append=False)

    # --- Train model ---
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[checkpoint, csv_logger]
    )

    # Evaluate on validation set
    print("\n--- Final evaluation on validation set ---\n")
    results = model.evaluate(val_ds)
    print("Validation results (loss, accuracy):", results)

    # Save final model
    final_model_path = os.path.join( "models", "resnet50_brain_tumor.h5")
    model.save(final_model_path)
    print(f"\n✅ Model saved to {final_model_path}")

if __name__ == "__main__":
    train(epochs=50,trainable_layers=200)
