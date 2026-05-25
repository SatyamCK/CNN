from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import os

app = FastAPI()

model = None

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

# Load model on startup
# Load model on startup
@app.on_event("startup")
async def load_model():

    global model

    try:

        print("===== STARTUP =====")

        print("Current directory:", os.getcwd())

        print("Root files:", os.listdir())

        if os.path.exists("model"):
            print("Model folder files:", os.listdir("model"))

        model_path = "model/cnn_clean.keras"

        print("Loading model from:", model_path)

        model = tf.keras.models.load_model(model_path)

        print("===== MODEL LOADED SUCCESSFULLY =====")

    except Exception as e:

        print("===== STARTUP ERROR =====")

        print(str(e))

        raise e

@app.get("/")
def home():
    return {"message": "CNN API Running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    global model

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