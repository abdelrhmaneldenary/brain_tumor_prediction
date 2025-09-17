from tensorflow.keras.applications import ResNet50
import tensorflow as tf
from tensorflow.keras import layers, models,regularizers
from tensorflow.keras.layers import Flatten,Dense,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from data import load_dataset

def build_model():
    # gpu_devices = tf.config.list_physical_devices('GPU')
    base_model=ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224,224,3)
    )
    #plot_model(base_model,to_file='base_model_arch.png') i runned it once 
    #freez the raining of the layers 
    for layer in base_model.layers[-30:]:
        layer.trainable=True

    
    model=models.Sequential([
        layers.Resizing(224,224),
        layers.Rescaling(1./255),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu',kernel_regularizer=regularizers.l2()
),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(4, activation='softmax',dtype='float32') ,   
        ])

    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model



if __name__ == "__main__":
    aug_ds, test_ds = load_dataset()
    model=build_model()
