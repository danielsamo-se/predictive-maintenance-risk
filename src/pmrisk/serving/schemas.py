from pydantic import BaseModel, Field, field_validator

from pmrisk.config import settings


class PredictRequest(BaseModel):
    window: list[dict[str, float]]

    @field_validator("window")
    @classmethod
    def validate_window_length(cls, v: list[dict[str, float]]) -> list[dict[str, float]]:
        if len(v) != settings.window_l:
            raise ValueError(f"window must have exactly {settings.window_l} rows, got {len(v)}")
        return v


class PredictResponse(BaseModel):
    risk_score: float = Field(ge=0.0, le=1.0)
