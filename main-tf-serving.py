from io import BytesIO
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
from PIL import Image
import tensorflow as tf
import os

app = FastAPI()

model_path = "C:/Users/DELL/Documents/AIProject/potateo-api/saved_models/1.keras"

if os.path.exists(model_path):
    print(f"Model found at {model_path}")
    MODEL = tf.keras.models.load_model(model_path)
else:
    raise FileNotFoundError(f"Model not found at {model_path}")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return {"message": "Pong!"}

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and preprocess the image
        image = read_file_as_image(await file.read())

        # Add batch dimension: shape becomes (1, 256, 256, 3)
        img_batch = np.expand_dims(image, axis=0)

        # Model prediction
        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
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
