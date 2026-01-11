"""IsolationForest for anomaly detection"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest


def fit_isoforest(X_train: np.ndarray, contamination: float, seed: int) -> IsolationForest:
    if X_train.ndim != 2:
        raise ValueError(f"X_train must be 2D, got shape {X_train.shape}")
    if not (0.0 < float(contamination) <= 0.5):
        raise ValueError(f"Invalid contamination={contamination}. Expected 0 < contamination <= 0.5")

    model = IsolationForest(
        contamination=float(contamination),
        random_state=int(seed),
        n_jobs=-1,
    )
    model.fit(X_train)
    return model


def score_isoforest(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")

    return (-model.score_samples(X)).astype(np.float64)