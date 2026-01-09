from fastapi.testclient import TestClient

from pmrisk.config import settings
from pmrisk.serving.app import app


def _make_zero_window(n_rows: int) -> list[dict[str, float]]:
    window: list[dict[str, float]] = []
    for _ in range(n_rows):
        row: dict[str, float] = {}
        for i in range(1, 4):
            row[f"op_setting_{i}"] = 0.0
        for i in range(1, 22):
            row[f"sensor_{i}"] = 0.0
        window.append(row)
    return window


def test_health_ok() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    data = response.json()

    assert "status" in data
    assert "model_version" in data
    assert "model_type" in data
    assert data["status"] == "ok"


def test_predict_valid_payload_returns_keys() -> None:
    payload = {"window": _make_zero_window(settings.window_l)}

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert "risk_score" in data
    assert "bucket" in data
    assert "threshold" in data
    assert "is_alert" in data
    assert "model_version" in data
    assert "model_type" in data

    assert data["bucket"] in {"low", "med", "high"}

    assert 0.0 <= data["risk_score"] <= 1.0
    assert data["bucket"] in {"low", "med", "high"}
    assert isinstance(data["is_alert"], bool)


def test_predict_wrong_length_422() -> None:
    payload = {"window": _make_zero_window(settings.window_l - 1)}

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_nan_422() -> None:
    window = _make_zero_window(settings.window_l)
    window[0]["sensor_5"] = "nan"  # JSON-safe "bad" value
    payload = {"window": window}

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422
