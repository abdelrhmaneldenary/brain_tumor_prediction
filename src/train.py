from data import load_dataset
from model import build_model
from tensorflow.keras.callbacks import CSVLogger
import tensorflow as tf
import os

def train():
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"   # optional: show TF logs
    # print("Physical GPUs:", tf.config.list_physical_devices('GPU'))

    # gpus = tf.config.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         for g in gpus:
    #             tf.config.experimental.set_memory_growth(g, True)
    #         print("Enabled memory growth for GPUs")
    #     except RuntimeError as e:
    #         print("Could not set memory growth:", e)


    aug_ds, test_ds = load_dataset()
    model = build_model()

    # Create a CSVLogger to save metrics during training
    csv_logger = CSVLogger("training_log.csv", append=False)

    history = model.fit(
        aug_ds,
        validation_data=test_ds,
        epochs=10,
        callbacks=[csv_logger]
    )

    # Save trained model
    model.save("models/resnet50_brain_tumor.h5")

if __name__ == "__main__":
    train()
