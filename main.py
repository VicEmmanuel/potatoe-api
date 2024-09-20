from io import BytesIO

from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import requests

app = FastAPI()

# model_path = "C:/Users/DELL/Documents/AIProject/potateo-api/saved_models/1.keras"
model_path = "saved_models/1.keras"

if os.path.exists(model_path):
    print(f"Model found at {model_path}")
    MODEL = tf.keras.models.load_model(model_path)
else:
    print(f"Model not found at {model_path}")

# MODEL = tf.keras.models.load_model("..saved_models/1.keras")
# MODEL = tf.keras.models.load_model("C:/Users/DELL/Documents/AIProject/saved_models/1.keras")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
@app.get("/ping")
async def ping():
    return "Hello, New"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image,0)
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }
    pass


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
