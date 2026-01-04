from fastapi.testclient import TestClient
from pmrisk.config import settings

from pmrisk.serving.app import app

client = TestClient(app)


def test_health_ok() -> None:
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "status" in data
    assert "model_version" in data
    assert "model_type" in data
    assert data["status"] == "ok"


def test_predict_valid_payload_returns_keys() -> None:
    window = []
    for _ in range(settings.window_l):
        row = {}
        for i in range(1, 4):
            row[f"op_setting_{i}"] = 0.0
        for i in range(1, 22):
            row[f"sensor_{i}"] = 0.0
        window.append(row)
    
    payload = {"window": window}
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "risk_score" in data
    assert "threshold" in data
    assert "is_alert" in data
    assert "model_version" in data
    assert "model_type" in data
    
    assert 0.0 <= data["risk_score"] <= 1.0
    assert isinstance(data["is_alert"], bool)


def test_predict_wrong_length_422() -> None:
    window = []
    for _ in range(settings.window_l - 1):
        row = {}
        for i in range(1, 4):
            row[f"op_setting_{i}"] = 0.0
        for i in range(1, 22):
            row[f"sensor_{i}"] = 0.0
        window.append(row)
    
    payload = {"window": window}
    response = client.post("/predict", json=payload)
    
    assert response.status_code in (400, 422)


def test_predict_nan_422() -> None:
    window = []
    for _ in range(settings.window_l):
        row = {}
        for i in range(1, 4):
            row[f"op_setting_{i}"] = 0.0
        for i in range(1, 22):
            row[f"sensor_{i}"] = 0.0
        window.append(row)
    
    window[0]["sensor_5"] = float("nan")
    
    payload = {"window": window}
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 422
