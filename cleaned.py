import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models

print("TensorFlow version:", tf.__version__)

# Compatibility patch
class CustomBatchNormalization(tf.keras.layers.BatchNormalization):
    def __init__(self, *args, renorm=False,
                 renorm_clipping=None,
                 renorm_momentum=0.99,
                 **kwargs):
        super().__init__(*args, **kwargs)

# Load old incompatible model
old_model = load_model(
    "model/Enhanced_cnn_cifar10_model1.h5",
    custom_objects={
        "BatchNormalization": CustomBatchNormalization
    },
    compile=False
)

print("Old model loaded!")

# Create clean model architecture
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

# Transfer weights
new_model.set_weights(old_model.get_weights())

print("Weights transferred!")

# Save clean modern model
new_model.save("model/cnn_clean.keras")

print("Clean model saved successfully!")