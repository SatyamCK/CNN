import tensorflow as tf

# Load old model
model = tf.keras.models.load_model(
    "model/Enhanced_cnn_cifar10_model1.keras",
    compile=False
)

print("Old model loaded")

# Create new clean model
new_model = tf.keras.Sequential()

# Keep only inference-safe layers
skip_layers = (
    tf.keras.layers.RandomFlip,
    tf.keras.layers.RandomRotation,
    tf.keras.layers.RandomZoom,
    tf.keras.layers.RandomContrast,
)

for layer in model.layers:
    if isinstance(layer, skip_layers):
        print(f"Skipping augmentation layer: {layer.name}")
        continue

    new_model.add(layer)

# Build model
new_model.build((None, 32, 32, 3))

print("New model built")

# Save cleaned model
new_model.save("model/cnn_fixed.keras")

print("Fixed model saved successfully!")