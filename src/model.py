from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from data import load_dataset

def build_model(input_shape=(224,224,3), l2=1e-4, dropout_rate=0.5, classes=4, trainable_layers=4):
    """
    ResNet50 transfer learning model with partial fine-tuning.
    Only the last `trainable_layers` are unfrozen.
    """

    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    # Freeze all layers first
    base_model.trainable = False

    # Unfreeze only the last `trainable_layers`
    for layer in base_model.layers[-trainable_layers:]:
        layer.trainable = True

    model = models.Sequential([
        layers.Resizing(input_shape[0], input_shape[1]),
        layers.Rescaling(1.0/255.0),
        base_model,
        layers.GlobalAveragePooling2D(),
        Dense(256, activation='relu', kernel_regularizer=regularizers.l2(l2)),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(classes, activation='softmax', dtype='float32'),
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


if __name__ == "__main__":
    train_ds, val_ds= load_dataset()
    m = build_model(trainable_layers=6)  # try 4, 6, or 8
    m.summary()
