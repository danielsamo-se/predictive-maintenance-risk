"""Error analysis utilities for false positives and false negatives"""

import pandas as pd


def summarize_errors(
    df: pd.DataFrame,
    horizon_n: int,
    early_alarm_slack: int,
    top_k: int = 20,
) -> dict:
    """Return top-K FPs/FNs sorted by y_proba descending."""
    required_cols = ["engine_id", "cycle", "label", "y_proba", "y_pred"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    has_remaining = "remaining" in df.columns
    df = df.copy()
    
    fp_df = df[(df["y_pred"] == 1) & (df["label"] == 0)].copy()
    fn_df = df[(df["y_pred"] == 0) & (df["label"] == 1)].copy()
    
    fp_df = fp_df.sort_values(
        ["y_proba", "engine_id", "cycle"],
        ascending=[False, True, True]
    ).head(top_k)
    
    fn_df = fn_df.sort_values(
        ["y_proba", "engine_id", "cycle"],
        ascending=[False, True, True]
    ).head(top_k)
    
    false_positives = []
    for _, row in fp_df.iterrows():
        entry = {
            "engine_id": int(row["engine_id"]),
            "cycle": int(row["cycle"]),
            "y_proba": float(row["y_proba"]),
            "label": int(row["label"]),
            "y_pred": int(row["y_pred"]),
        }
        if has_remaining and pd.notna(row["remaining"]):
            rem = int(row["remaining"])
            entry["remaining"] = rem
            
            if 0 < rem <= horizon_n + early_alarm_slack:
                entry["fp_kind"] = "premature_alarm"
            elif rem <= 0:
               entry["fp_kind"] = "at_or_past_failure"
            else:
                entry["fp_kind"] = "noise_fp"
        false_positives.append(entry)
    
    false_negatives = []
    for _, row in fn_df.iterrows():
        entry = {
            "engine_id": int(row["engine_id"]),
            "cycle": int(row["cycle"]),
            "y_proba": float(row["y_proba"]),
            "label": int(row["label"]),
            "y_pred": int(row["y_pred"]),
        }
        if has_remaining and pd.notna(row["remaining"]):
            entry["remaining"] = int(row["remaining"])
        false_negatives.append(entry)
    
    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }
