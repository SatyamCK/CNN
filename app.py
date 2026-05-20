from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import os

print("Starting application...")

app = FastAPI()

# MODEL PATH
MODEL_PATH = "model/cnn_fixed.keras"

print("Checking model path:", MODEL_PATH)
print("Current working directory:", os.getcwd())

# Check model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

print("Loading model...")

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# Classes
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

@app.get("/")
def home():
    return {"message": "CNN API Running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()

    image = Image.open(io.BytesIO(contents)).convert("RGB")

    image = image.resize((32, 32))

    image_array = np.array(image) / 255.0

    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array)

    predicted_class = class_names[np.argmax(predictions)]

    confidence = float(np.max(predictions))

    return {
        "prediction": predicted_class,
        "confidence": round(confidence * 100, 2)
    }

# IMPORTANT FOR RENDER
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(app, host="0.0.0.0", port=port)