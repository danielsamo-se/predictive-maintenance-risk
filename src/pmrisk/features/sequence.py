"""Sequence feature extraction for time-series models"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def get_sequence_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("op_setting_") or c.startswith("sensor_")]
    return sorted(cols)


def fit_standard_scaler_params(
    df: pd.DataFrame,
    feature_columns: list[str],
    train_engine_ids: list[int],
) -> dict[str, list[float] | list[str]]:
    if not feature_columns:
        raise ValueError("feature_columns must not be empty")

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in df: {missing}")

    if "engine_id" not in df.columns:
        raise ValueError("df must contain 'engine_id'")

    train_ids = set(int(x) for x in train_engine_ids)
    train_df = df[df["engine_id"].astype(int).isin(train_ids)]

    if train_df.empty:
        raise ValueError("No rows found for train_engine_ids")

    x = train_df[feature_columns].to_numpy(dtype=float)
    mean = x.mean(axis=0)
    std = x.std(axis=0, ddof=0)

    std = np.where(std == 0.0, 1.0, std)

    return {
        "feature_columns": list(feature_columns),
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std],
    }


def apply_standard_scaler(df: pd.DataFrame, scaler_params: dict[str, Any]) -> pd.DataFrame:
    if "feature_columns" not in scaler_params or "mean" not in scaler_params or "std" not in scaler_params:
        raise ValueError("scaler_params must contain feature_columns, mean, std")

    feature_columns = list(scaler_params["feature_columns"])
    mean = np.asarray(scaler_params["mean"], dtype=float)
    std = np.asarray(scaler_params["std"], dtype=float)

    if len(feature_columns) != len(mean) or len(feature_columns) != len(std):
        raise ValueError("scaler_params lengths mismatch (feature_columns/mean/std)")

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in df: {missing}")

    df_out = df.copy()

    df_out[feature_columns] = df_out[feature_columns].astype("float64")

    x = df_out[feature_columns].to_numpy(dtype="float64", copy=False)
    x_scaled = (x - mean) / std

    df_out[feature_columns] = x_scaled
    return df_out


def make_engine_arrays(
    df: pd.DataFrame,
    feature_columns: list[str],
    label_col: str = "label",
) -> dict[int, dict[str, np.ndarray]]:
    required = ["engine_id", "cycle", label_col]
    missing_req = [c for c in required if c not in df.columns]
    if missing_req:
        raise ValueError(f"Missing required columns: {missing_req}")

    missing_feat = [c for c in feature_columns if c not in df.columns]
    if missing_feat:
        raise ValueError(f"Missing feature columns in df: {missing_feat}")

    df_sorted = df.sort_values(["engine_id", "cycle"], kind="mergesort").reset_index(drop=True)
    has_remaining = "remaining" in df_sorted.columns

    out: dict[int, dict[str, np.ndarray]] = {}
    for engine_id, g in df_sorted.groupby("engine_id", sort=True):
        engine_id_int = int(engine_id)
        x = g[feature_columns].to_numpy(dtype=np.float32)
        y = g[label_col].to_numpy(dtype=np.int64)
        cycles = g["cycle"].to_numpy(dtype=np.int64)

        d: dict[str, np.ndarray] = {"X": x, "y": y, "cycles": cycles}
        if has_remaining:
            d["remaining"] = g["remaining"].to_numpy(dtype=np.float32)

        out[engine_id_int] = d

    return out


def make_sequence_index(
    engine_arrays: dict[int, dict[str, np.ndarray]],
    window_length: int,
) -> list[dict]:
    if window_length < 1:
        raise ValueError("window_length must be >= 1")

    index: list[dict] = []
    for engine_id in sorted(engine_arrays.keys()):
        cycles = engine_arrays[engine_id]["cycles"]
        t = int(len(cycles))
        for end_pos in range(window_length - 1, t):
            index.append(
                {
                    "engine_id": int(engine_id),
                    "pred_cycle": int(cycles[end_pos]),
                    "end_pos": int(end_pos),
                }
            )
    return index


def extract_window(
    engine_arrays: dict[int, dict[str, np.ndarray]],
    engine_id: int,
    end_pos: int,
    window_length: int,
) -> np.ndarray:
    if window_length < 1:
        raise ValueError("window_length must be >= 1")
    if engine_id not in engine_arrays:
        raise KeyError(f"engine_id not found: {engine_id}")

    x = engine_arrays[engine_id]["X"]
    t = x.shape[0]

    if end_pos < 0 or end_pos >= t:
        raise ValueError(f"end_pos out of range: {end_pos} (T={t})")

    start_pos = end_pos - window_length + 1
    if start_pos < 0:
        raise ValueError("end_pos must be >= window_length - 1")

    window = x[start_pos : end_pos + 1]
    if window.shape[0] != window_length:
        raise ValueError("Extracted window has wrong length")

    return window.astype(np.float32, copy=False)
