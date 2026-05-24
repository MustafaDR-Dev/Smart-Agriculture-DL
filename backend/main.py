import io
import numpy as np
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow_datasets as tfds

IMG_SIZE = 224
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "mobilenetv2_best.keras"

info = tfds.builder("plant_village").info
CLASS_NAMES = info.features["label"].names


# Schemas de reponse
class PredictionItem(BaseModel):
    label: str
    confidence: float


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    top5: list[PredictionItem]


# Meme preprocess que pendant l'entrainement (MobileNetV2)
def preprocess_image(raw_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = preprocess_input(np.array(img, dtype=np.float32))
    return np.expand_dims(arr, axis=0)


def fmt_label(name: str) -> str:
    plant, disease = name.split("___")
    return plant.replace("_", " ") + " - " + disease.replace("_", " ")


app = FastAPI(title="Plant Disease API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = tf.keras.models.load_model(str(MODEL_PATH))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    img = preprocess_image(data)

    preds = model.predict(img, verbose=0)[0]
    best = int(np.argmax(preds))
    top5 = np.argsort(preds)[::-1][:5]

    return {
        "prediction": fmt_label(CLASS_NAMES[best]),
        "confidence": round(float(preds[best]), 4),
        "top5": [{"label": fmt_label(CLASS_NAMES[i]), "confidence": round(float(preds[i]), 4)} for i in top5],
    }
