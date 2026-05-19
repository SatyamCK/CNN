import tensorflow as tf

# Load existing h5 model
model = tf.keras.models.load_model(
    "model/Enhanced_cnn_cifar10_model1.h5",
    compile=False
)

# Save in new keras format
model.save("model/Enhanced_cnn_cifar10_model1.keras")

print("Model converted successfully!")