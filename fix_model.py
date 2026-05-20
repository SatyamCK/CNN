import tensorflow as tf

print("TensorFlow Version:", tf.__version__)

# Load old model
model = tf.keras.models.load_model(
    "model/Enhanced_cnn_cifar10_model1.h5",
    compile=False
)

print("Model loaded successfully!")

# Save in new format
model.save("model/cnn_clean.keras")

print("Model exported successfully!")