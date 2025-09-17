import tensorflow as tf 
import os
import kagglehub
from pathlib import Path
from tensorflow.keras import layers
AUTOTUNE = tf.data.AUTOTUNE



def load_dataset():
    #gpu_devices = tf.config.list_physical_devices('GPU')
    # if gpu_devices:
    #     print(f"✅ GPU is available and TensorFlow is using it: {gpu_devices[0].name}")
    # else:
    #     print("❌ GPU not found. TensorFlow is using the CPU.")

    path_str = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
    path=Path(path_str)
    train_path=Path(os.path.join(path,'Training'))
    test_path=Path(os.path.join(path,'Testing'))
    # i used the  below code to ensure that the data is in its place
    # image_count=len(list(path.glob('*/*/*.jpg')))
    # train_count=len(list(train_data.glob('*/*.jpg')))
    # test_count=len(list(test_data.glob('*/*.jpg')))
    # print(image_count)
    # print(train_count)
    # print(test_count)
    batch_size=32
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
    train_path,
    label_mode='categorical',
    seed=123,
    batch_size=batch_size)

    test_ds=tf.keras.utils.image_dataset_from_directory(
        test_path,
        label_mode='categorical',
        seed=123,
        batch_size=batch_size
    )

    data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(factor=(20/360)),
    layers.RandomWidth(factor=0.08),
    layers.RandomHeight(factor=0.08),
    layers.RandomZoom(height_factor=0.08),   
        ], name="data_augmentation")
    
    aug_ds = train_ds.map(
    lambda image, label: (data_augmentation(image, training=True), label),
    num_parallel_calls=AUTOTUNE
        ).prefetch(AUTOTUNE)

    test_ds = test_ds.prefetch(AUTOTUNE)

    return aug_ds,test_ds

if __name__ =="__main__":
    load_dataset()
