"""Sequence model predictor"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from pmrisk.models.model_builder import build_sequence_model
from pmrisk.models.model_versions import get_active_version, load_torch_state_dict, read_metadata


class SequencePredictor:
    """Loads the active sequence model and runs inference with train-time scaling."""

    def __init__(self, root: Path | None = None) -> None:
        if root is None:
            root = Path("models") / "production"

        model_name = "seq"
        self.version = get_active_version(model_name, root=root)
        version_dir = root / model_name / self.version

        self.metadata = read_metadata(version_dir)
        state_dict = load_torch_state_dict(version_dir)

        hparams = self.metadata["hparams"]
        self.model = build_sequence_model(hparams)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.window_l = int(hparams["window_l"])
        self.n_features = int(hparams["n_features"])
        self.threshold = float(self.metadata["threshold"])
        self.model_type = str(hparams["model_type"])

        sp = self.metadata["scaler_params"]
        self.scaler_mean = np.array(sp["mean"], dtype=np.float32)
        self.scaler_std = np.array(sp["std"], dtype=np.float32)
        self.feature_columns = list(sp["feature_columns"])

        # Policy for buckets (low/med/high). Prefer metadata, fallback for older models.
        cutoffs = self.metadata.get("bucket_cutoffs", [0.2, 0.5])
        if not isinstance(cutoffs, list) or len(cutoffs) != 2:
            raise ValueError("metadata bucket_cutoffs must be a list with exactly 2 values")
        self.bucket_cutoffs = [float(cutoffs[0]), float(cutoffs[1])]

    def predict(self, window: list[dict[str, float]]) -> dict:
        """Predict risk_score + policy outputs for one window."""
        if len(window) != self.window_l:
            raise ValueError(f"window length must be {self.window_l}, got {len(window)}")

        x_rows: list[list[float]] = []
        for i, row in enumerate(window):
            missing = [c for c in self.feature_columns if c not in row]
            if missing:
                raise ValueError(f"Row {i} missing keys: {missing}")
            x_rows.append([float(row[c]) for c in self.feature_columns])

        x = np.asarray(x_rows, dtype=np.float32)

        if x.shape != (self.window_l, self.n_features):
            raise ValueError(
                f"window must have shape ({self.window_l}, {self.n_features}), got {x.shape}"
            )

        if not np.isfinite(x).all():
            raise ValueError("All window values must be finite (no NaN/Inf)")

        x_scaled = (x - self.scaler_mean) / self.scaler_std
        x_tensor = torch.from_numpy(x_scaled).unsqueeze(0)  # (1, L, F)

        with torch.no_grad():
            logits = self.model(x_tensor)
            risk_score = float(torch.sigmoid(logits).item())

        is_alert = risk_score >= self.threshold

        low_med, med_high = self.bucket_cutoffs
        if risk_score < low_med:
            bucket = "low"
        elif risk_score < med_high:
            bucket = "med"
        else:
            bucket = "high"

        return {
            "risk_score": risk_score,
            "bucket": bucket,
            "threshold": self.threshold,
            "is_alert": is_alert,
            "model_version": self.version,
            "model_type": self.model_type,
        }