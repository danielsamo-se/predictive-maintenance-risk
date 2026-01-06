"""FastAPI app for model serving"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from pmrisk.serving.predictor import SequencePredictor
from pmrisk.serving.schemas import PredictRequest, PredictResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    models_root = os.getenv("PMRISK_MODELS_ROOT", "models/production")
    try:
        app.state.predictor = SequencePredictor(root=Path(models_root))
    except Exception as e:
        print(f"Warning: predictor not loaded: {e}")
        app.state.predictor = None
    yield


app = FastAPI(title="PM Risk Predictor", lifespan=lifespan)


@app.get("/health")
def health():
    p = getattr(app.state, "predictor", None)
    if p is None:
        raise HTTPException(status_code=503, detail={"status": "not_ready"})

    return {"status": "ok", "model_version": p.version, "model_type": p.model_type}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    p = getattr(app.state, "predictor", None)
    if p is None:
        raise HTTPException(status_code=503, detail="predictor not ready")

    try:
        result = p.predict(request.window)
        return PredictResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
