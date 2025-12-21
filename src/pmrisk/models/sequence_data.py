"""Builds scaled sequences and an index per engine"""

import numpy as np
import pandas as pd

from pmrisk.features.sequence import (
    apply_standard_scaler,
    extract_window,
    fit_standard_scaler_params,
    get_sequence_feature_columns,
    make_engine_arrays,
    make_sequence_index,
)


def build_sequence_pipeline(
    df: pd.DataFrame,
    train_engine_ids: list[int],
    window_length: int,
    label_col: str = "label",
) -> dict:
    
    feature_columns = get_sequence_feature_columns(df)
    
    if not feature_columns:
        raise ValueError("No feature columns found (must start with op_setting_ or sensor_)")
    
    scaler_params = fit_standard_scaler_params(df, feature_columns, train_engine_ids)
    
    df_scaled = apply_standard_scaler(df, scaler_params)

    x_scaled = df_scaled[feature_columns].to_numpy(dtype=float)
    if not np.isfinite(x_scaled).all():
        raise ValueError("Non-finite values found after scaling")
    
    engine_arrays = make_engine_arrays(df_scaled, feature_columns, label_col=label_col)
    index = make_sequence_index(engine_arrays, window_length)
    
    if not index:
        raise ValueError("No valid windows in index (check window_length vs cycles)")

    first = index[0]
    sample_window = extract_window(
        engine_arrays,
        first["engine_id"],
        first["end_pos"],
        window_length,
    )
    
    expected_shape = (window_length, len(feature_columns))
    if sample_window.shape != expected_shape:
        raise ValueError(f"Sample window shape {sample_window.shape} != expected {expected_shape}")
    
    if not np.isfinite(sample_window).all():
        raise ValueError("Sample window contains non-finite values")
    
    return {
        "feature_columns": feature_columns,
        "scaler_params": scaler_params,
        "engine_arrays": engine_arrays,
        "index": index,
    }
