from io import BytesIO
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
from PIL import Image
import tensorflow as tf
import os

app = FastAPI()

# Update model path if needed
model_path = "C:/Users/DELL/Documents/AIProject/potateo-api/saved_models/pneumonia.keras"

if os.path.exists(model_path):
    print(f"Model found at {model_path}")
    MODEL = tf.keras.models.load_model(model_path)
else:
    raise FileNotFoundError(f"Model not found at {model_path}")

CLASS_NAMES = ["Normal", "Pneumonia"]

@app.get("/ping")
async def ping():
    return {"message": "Pong!"}

def read_file_as_image(data: bytes) -> np.ndarray:
    image = Image.open(BytesIO(data))
    image = image.convert("RGB")  # Ensure image is in RGB mode
    return np.array(image)

def preprocess_image(image: np.ndarray, target_size=(256, 256)) -> np.ndarray:
    # Resize image
    image = Image.fromarray(image)
    image = image.resize(target_size, Image.LANCZOS)
    image = np.array(image)

    # Ensure image has 3 channels
    if image.shape[-1] != 3:
        raise ValueError("Image must have 3 color channels (RGB).")

    # Normalize image to [0, 1]
    image = image / 255.0

    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and preprocess the image
        image = read_file_as_image(await file.read())
        image = preprocess_image(image)

        # Add batch dimension: shape becomes (1, 256, 256, 3)
        img_batch = np.expand_dims(image, axis=0)

        # Model prediction
        predictions = MODEL.predict(img_batch)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "Prediction made successfully",
                "data": {
                    "class": predicted_class,
                    "confidence": confidence
                }
            }
        )
    except Exception as e:
        # Return failure response with structured data
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "An error occurred during prediction",
                "data": str(e)
            }
        )

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
