"""Seq predictor: load active model + scale window + return risk score"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from pmrisk.models.model_builder import build_sequence_model
from pmrisk.models.model_versions import get_active_version, load_torch_state_dict, read_metadata


class SequencePredictor:
    """Loads the active seq model and runs inference"""

    def __init__(self, root: Path | None = None) -> None:
        if root is None:
            root = Path("models") / "production"

        model_name = "seq"
        self.version = get_active_version(model_name, root=root)
        self.version_dir = root / model_name / self.version

        self.metadata = read_metadata(self.version_dir)
        state_dict = load_torch_state_dict(self.version_dir)

        hparams = self.metadata["hparams"]
        self.window_l = int(hparams["window_l"])
        self.n_features = int(hparams["n_features"])
        self.model_type = str(hparams["model_type"])
        self.threshold = float(self.metadata["threshold"])

        sp: dict[str, Any] = self.metadata["scaler_params"]
        self.feature_columns = list(sp["feature_columns"])
        self.mean = np.asarray(sp["mean"], dtype=np.float32)
        self.std = np.asarray(sp["std"], dtype=np.float32)

        if len(self.feature_columns) != self.n_features:
            raise ValueError("metadata mismatch: feature_columns length != n_features")
        if self.mean.shape != (self.n_features,) or self.std.shape != (self.n_features,):
            raise ValueError("metadata mismatch: mean/std shape != (n_features,)")

        self.model = build_sequence_model(hparams)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, window: list[dict[str, float]]) -> dict:
        """Predict risk score for one window (list of dict rows)."""
        if len(window) != self.window_l:
            raise ValueError(f"window must have {self.window_l} rows, got {len(window)}")

        rows: list[list[float]] = []
        for i, row in enumerate(window):
            missing = [c for c in self.feature_columns if c not in row]
            if missing:
                raise ValueError(f"row {i} missing keys: {missing}")
            rows.append([float(row[c]) for c in self.feature_columns])

        x = np.asarray(rows, dtype=np.float32)

        if x.shape != (self.window_l, self.n_features):
            raise ValueError(f"window must be shape ({self.window_l},{self.n_features}), got {x.shape}")
        if not np.isfinite(x).all():
            raise ValueError("window contains NaN/Inf")

        x_scaled = (x - self.mean) / self.std
        x_tensor = torch.from_numpy(x_scaled).unsqueeze(0) 

        with torch.no_grad():
            logits = self.model(x_tensor)
            risk_score = float(torch.sigmoid(logits).reshape(-1)[0].item())

        return {
            "risk_score": risk_score,
            "threshold": self.threshold,
            "is_alert": risk_score >= self.threshold,
            "model_version": self.version,
            "model_type": self.model_type,
        }
