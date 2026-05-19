from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import tensorflow as tf
import io

app = FastAPI()

# Load trained model
model = tf.keras.models.load_model("model/Enhanced_cnn_cifar10_model1.keras")

# CIFAR10 Classes
class_names = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

# Home route
@app.get("/")
def home():
    return {
        "message": "CIFAR10 Image Classification API Running"
    }

# Prediction route
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read image
    contents = await file.read()

    # Convert to PIL image
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Resize image to CIFAR10 input size
    image = image.resize((32, 32))

    # Convert image to numpy array
    image_array = np.array(image)

    # Normalize
    image_array = image_array / 255.0

    # Expand dimensions
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    predictions = model.predict(image_array)

    predicted_class = class_names[np.argmax(predictions)]

    confidence = float(np.max(predictions))

    return {
        "prediction": predicted_class,
        "confidence": round(confidence * 100, 2)
    }