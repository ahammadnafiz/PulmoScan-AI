import base64
import io

from fastapi.testclient import TestClient
from PIL import Image

from app.services.inference import inference_service


def _fake_predict(image_bytes: bytes) -> dict:
    # Decode to validate the bytes are a real image, then return a fixed result.
    Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return {
        "label": "normal",
        "confidence": 0.91,
        "probabilities": [
            {"label": "adenocarcinoma", "probability": 0.09},
            {"label": "normal", "probability": 0.91},
        ],
        "inference_time_ms": 4.2,
    }


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color=(120, 120, 120)).save(buf, format="PNG")
    return buf.getvalue()


def test_predict_requires_loaded_model(client: TestClient, monkeypatch):
    # Model is not loaded in tests -> 503.
    monkeypatch.setattr(inference_service, "_models", [])
    files = {"file": ("scan.png", _png_bytes(), "image/png")}
    response = client.post("/api/v1/predict", files=files)
    assert response.status_code == 503


def test_predict_upload(client: TestClient, monkeypatch):
    monkeypatch.setattr(inference_service, "_models", [object()])  # mark as ready
    monkeypatch.setattr(inference_service, "predict", _fake_predict)
    files = {"file": ("scan.png", _png_bytes(), "image/png")}
    response = client.post("/api/v1/predict", files=files)
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "normal"
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["probabilities"]) == 2


def test_predict_rejects_non_image(client: TestClient, monkeypatch):
    monkeypatch.setattr(inference_service, "_models", [object()])
    files = {"file": ("notes.txt", b"hello", "text/plain")}
    response = client.post("/api/v1/predict", files=files)
    assert response.status_code == 422


def test_predict_base64(client: TestClient, monkeypatch):
    monkeypatch.setattr(inference_service, "_models", [object()])
    monkeypatch.setattr(inference_service, "predict", _fake_predict)
    payload = {"image": base64.b64encode(_png_bytes()).decode()}
    response = client.post("/api/v1/predict/base64", json=payload)
    assert response.status_code == 200
    assert response.json()["label"] == "normal"
