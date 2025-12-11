"""
Calibration utilities (Brier score + per-bin stats)
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss


def compute_calibration(
    y_true,
    y_proba,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> dict:
    """Return Brier score and calibration bins"""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    if len(y_true) != len(y_proba):
        raise ValueError(
            f"y_true and y_proba must have same length, got {len(y_true)} and {len(y_proba)}"
        )
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    if strategy not in {"uniform", "quantile"}:
        raise ValueError(f"strategy must be 'uniform' or 'quantile', got {strategy}")
    if np.any(y_proba < 0) or np.any(y_proba > 1):
        raise ValueError("y_proba must be in [0, 1]")

    brier = float(brier_score_loss(y_true, y_proba))

    if strategy == "uniform":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        q = np.linspace(0.0, 1.0, n_bins + 1)
        bin_edges = np.quantile(y_proba, q)
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            bin_edges = np.array([0.0, 1.0])

    bin_idx = np.digitize(y_proba, bin_edges[1:-1], right=True)
    n_effective_bins = max(1, len(bin_edges) - 1)

    bins = []
    for i in range(n_effective_bins):
        mask = bin_idx == i
        count = int(mask.sum())
        if count == 0:
            continue

        bins.append(
            {
                "bin_idx": int(i),
                "count": count,
                "mean_pred": float(np.mean(y_proba[mask])),
                "frac_pos": float(np.mean(y_true[mask])),
            }
        )

    return {"brier_score": brier, "bins": bins}
