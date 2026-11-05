
from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from src.api import main


class FakeModel:
    """Modelo simulado para probar la API sin cargar TensorFlow."""

    def predict(self, image_array, verbose=0):
        return np.array([[0.05, 0.10, 0.05, 0.10, 0.65, 0.05]])


def create_test_image():
    """Genera una imagen JPEG válida en memoria."""
    image = Image.new("RGB", (224, 224), color="white")
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer


def test_home():
    response = TestClient(main.app).get("/")

    assert response.status_code == 200


def test_health():
    main.model = FakeModel()

    response = TestClient(main.app).get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_predict_valid_image():
    main.model = FakeModel()

    response = TestClient(main.app).post(
        "/predict",
        files={
            "file": (
                "imagen.jpg",
                create_test_image(),
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["predicted_class"] == "Benign"
    assert "confidence" in data
    assert "probabilities" in data
    assert len(data["probabilities"]) == 6


def test_predict_invalid_content_type():
    main.model = FakeModel()

    response = TestClient(main.app).post(
        "/predict",
        files={
            "file": (
                "documento.txt",
                b"esto no es una imagen",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400
