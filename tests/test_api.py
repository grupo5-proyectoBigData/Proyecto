
from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from src.api import main


# Modelo simulado para probar la API sin cargar VGG16
class FakeModel:
    def predict(self, image, verbose=0):
        return np.array([[0.05, 0.05, 0.05, 0.05, 0.10, 0.70]])


client = TestClient(main.app)


def create_test_image():
    image = Image.new("RGB", (224, 224), color="red")
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["model"] == "VGG16"


def test_health():
    main.model = FakeModel()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_predict_valid_image(monkeypatch):
    monkeypatch.setattr(main, "model", FakeModel())

    image = create_test_image()

    response = client.post(
        "/predict",
        files={"file": ("test.jpg", image, "image/jpeg")}
    )

    assert response.status_code == 200
    assert response.json()["model"] == "VGG16"
    assert response.json()["predicted_class"] == "6. Malign"
    assert response.json()["confidence"] == 0.70


def test_predict_invalid_content_type(monkeypatch):
    monkeypatch.setattr(main, "model", FakeModel())

    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"texto", "text/plain")}
    )

    assert response.status_code == 400
