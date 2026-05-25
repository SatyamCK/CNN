import tensorflow as tf
from tensorflow.keras import layers, models

print("TensorFlow:", tf.__version__)

# Build NEW clean architecture
new_model = models.Sequential([

    layers.Input(shape=(32, 32, 3)),

    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),

    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),

    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),

    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Flatten(),

    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(10, activation='softmax')
])

print("New architecture created!")

# LOAD WEIGHTS ONLY
new_model.load_weights(
    "Enhanced_cnn_cifar10_model1.h5",
    by_name=True,
    skip_mismatch=True
)

print("Weights loaded!")

# SAVE CLEAN MODEL
new_model.save("model/cnn_final.keras")

print("Final clean model saved!")