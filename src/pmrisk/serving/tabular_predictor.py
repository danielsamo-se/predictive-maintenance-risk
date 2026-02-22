"""Tabular model predictor – computes features from a raw sensor window"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml


class TabularPredictor:
    """Loads the tabular model (joblib) and runs inference on a single window."""

    def __init__(self, root: Path | None = None) -> None:
        if root is None:
            root = Path("models") / "production"

        metadata_path = root / "metadata.json"
        model_path = root / "model.joblib"

        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata not found: {metadata_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"model not found: {model_path}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.model = joblib.load(model_path)
        self.model_type = self.metadata["model_type"]
        self.version = self.metadata.get("featureset_version", "v1")
        self.feature_columns = self.metadata["feature_columns"]
        self.threshold = float(self.metadata["threshold"])

        cutoffs = self.metadata.get("bucket_cutoffs", [0.2, 0.5])
        self.bucket_cutoffs = [float(cutoffs[0]), float(cutoffs[1])]

        features_path = Path("configs/features.yaml")
        with open(features_path, "r", encoding="utf-8") as f:
            self.feat_cfg = yaml.safe_load(f)

    def _compute_features(self, window: list[dict[str, float]]) -> np.ndarray:
        """Compute tabular features from a raw sensor window (last row only)."""
        df = pd.DataFrame(window)

        signal_cols = sorted([
            c for c in df.columns
            if c.startswith(("op_setting_", "sensor_"))
        ])

        features = {}

        for col in signal_cols:
            values = df[col].to_numpy()

            for w in self.feat_cfg["windows"]:
                chunk = values[-w:]
                for stat in self.feat_cfg["stats"]:
                    feat_name = f"{col}__roll{w}__{stat}"
                    if stat == "mean":
                        features[feat_name] = float(np.mean(chunk))
                    elif stat == "std":
                        features[feat_name] = float(np.std(chunk, ddof=1)) if len(chunk) > 1 else 0.0

            for lag in self.feat_cfg["lags"]:
                feat_name = f"{col}__lag{lag}"
                features[feat_name] = float(values[-(lag + 1)])

            for delta in self.feat_cfg["deltas"]:
                feat_name = f"{col}__delta{delta}"
                features[feat_name] = float(values[-1] - values[-(delta + 1)])

        row = [features[c] for c in self.feature_columns]
        return np.array([row], dtype=np.float64)

    def predict(self, window: list[dict[str, float]]) -> dict:
        """Predict risk_score + policy outputs for one window."""
        X = self._compute_features(window)

        if not np.isfinite(X).all():
            raise ValueError("computed features contain NaN/Inf")

        risk_score = float(self.model.predict_proba(X)[0, 1])
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
