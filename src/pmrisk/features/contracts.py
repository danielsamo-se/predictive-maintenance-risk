"""Validate tabular features: sorted/no dups/no NaNs after window_l"""
import pandas as pd
import numpy as np


def validate_tabular_features(
    df: pd.DataFrame,
    window_l: int,
    id_col: str = "engine_id",
    time_col: str = "cycle",
) -> None:
    if id_col not in df.columns or time_col not in df.columns:
        raise ValueError(f"Missing required columns: {id_col} or {time_col}")
    
    duplicates = df[df.duplicated(subset=[id_col, time_col], keep=False)]
    if not duplicates.empty:
        raise ValueError(f"Found duplicates on ({id_col}, {time_col})")
    
    if not df.equals(df.sort_values([id_col, time_col]).reset_index(drop=True)):
        raise ValueError(f"DataFrame not sorted by ({id_col}, {time_col})")
    
    feature_cols = [c for c in df.columns if c not in [id_col, time_col]]
    if not feature_cols:
        raise ValueError("No feature columns found")
    
    for col in feature_cols:
        if not np.all(np.isfinite(df[col].dropna())):
            raise ValueError(f"Feature column {col} contains non-finite values")
    
    df_filtered = df[df[time_col] >= window_l]
    for col in feature_cols:
        if df_filtered[col].isna().any():
            raise ValueError(f"Feature column {col} contains NaN after filtering cycle >= {window_l}")
