from fastapi.testclient import TestClient


def test_root(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "PulmoScan AI"
    assert body["docs"] == "/docs"


def test_health(client: TestClient):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"healthy", "degraded"}
    assert "model_loaded" in body
    assert "device" in body


def test_liveness(client: TestClient):
    response = client.get("/api/v1/health/live")
    assert response.status_code == 200
    assert response.json() == {"alive": True}
