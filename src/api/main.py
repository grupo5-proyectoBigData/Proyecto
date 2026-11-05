
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from io import BytesIO

# Rutas del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "src" / "models" / "mobilenetv2.keras"

# Clases del modelo
CLASS_NAMES = [
    "Enfeksiyonel",
    "Ezkama",
    "Akne",
    "Pigment",
    "Benign",
    "Malign",
]

IMAGE_SIZE = (224, 224)

# Inicializar API
app = FastAPI(
    title="API de clasificación de imágenes de piel",
    description="API para realizar predicciones mediante MobileNetV2",
    version="1.0.0",
)

# Cargar modelo
model = None


@app.on_event("startup")
def load_model():
    global model

    if not MODEL_PATH.exists():
        raise RuntimeError(f"No se encuentra el modelo: {MODEL_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)
    print("Modelo MobileNetV2 cargado correctamente")


@app.get("/")
def home():
    return {
        "mensaje": "API de clasificación funcionando",
        "modelo": "MobileNetV2",
        "estado": "activo",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo todavía no está cargado",
        )

    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=400,
            detail="Formato no permitido. Utiliza JPG, PNG o WEBP.",
        )

    try:
        contents = await file.read()

        image = Image.open(BytesIO(contents)).convert("RGB")
        image = image.resize(IMAGE_SIZE)

        image_array = np.asarray(image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_array, verbose=0)[0]

        predicted_index = int(np.argmax(predictions))

        return {
            "predicted_class": CLASS_NAMES[predicted_index],
            "confidence": float(predictions[predicted_index]),
            "probabilities": {
                CLASS_NAMES[i]: float(predictions[i])
                for i in range(len(CLASS_NAMES))
            },
        }

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"No se pudo procesar la imagen: {str(error)}",
        )
