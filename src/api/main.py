
from pathlib import Path
from io import BytesIO

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input

# Rutas
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "src" / "models" / "vgg16.keras"

# Clases en el orden del metadata.json
CLASS_NAMES = [
    "1. Enfeksiyonel",
    "2. Ekzama",
    "3. Akne",
    "4. Pigment",
    "5. Benign",
    "6. Malign",
]

IMAGE_SIZE = (224, 224)

app = FastAPI(
    title="API de clasificación dermatológica - VGG16",
    description="Prototipo académico de clasificación de imágenes en seis categorías.",
    version="2.0.0",
)

model = None


@app.on_event("startup")
def load_model():
    global model

    if not MODEL_PATH.exists():
        raise RuntimeError(f"No se encuentra el modelo: {MODEL_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)
    print("Modelo VGG16 cargado correctamente")


@app.get("/")
def home():
    return {
        "message": "API de clasificación dermatológica con VGG16",
        "model": "VGG16",
        "docs": "/docs",
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
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    if file.content_type not in [
        "image/jpeg",
        "image/png",
        "image/webp",
    ]:
        raise HTTPException(
            status_code=400,
            detail="Formato no permitido. Usa JPEG, PNG o WEBP.",
        )

    try:
        contents = await file.read()

        image = Image.open(BytesIO(contents)).convert("RGB")
        image = image.resize(IMAGE_SIZE)

        image_array = np.asarray(image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)

        # Preprocesamiento específico de VGG16
        image_array = preprocess_input(image_array)

        predictions = model.predict(image_array, verbose=0)[0]

        predicted_index = int(np.argmax(predictions))

        return {
            "model": "VGG16",
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
    finally:
        await file.close()
