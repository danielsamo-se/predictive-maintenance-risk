from pmrisk.config import settings
from pmrisk.serving.schemas import PredictRequest, PredictResponse


def test_smoke_imports():
    dummy_row = {"sensor_1": 1.0, "sensor_2": 2.0}
    request = PredictRequest(window=[dummy_row] * settings.window_l)
    assert len(request.window) == settings.window_l

    response = PredictResponse(
    risk_score=0.5,
    threshold=0.2,
    is_alert=False,
    model_version="v0-test",
    model_type="gru",
)
    assert 0.0 <= response.risk_score <= 1.0
