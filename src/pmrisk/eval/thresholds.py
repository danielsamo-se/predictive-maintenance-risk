"""
 Search over observed probabilities (plus 0/1) to keep selection deterministic
 Guardrails: require target recall and minimum precision; otherwise fail fast
"""

import numpy as np
from sklearn.metrics import precision_score, recall_score


def select_threshold(
    y_true,
    y_proba,
    target_recall: float,
    min_precision: float,
) -> float:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    
    thresholds = np.unique(y_proba)
    thresholds = np.sort(thresholds)
    thresholds = np.concatenate([[0.0], thresholds, [1.0]])
    
    candidates = []
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        
        rec = recall_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        
        if rec >= target_recall and prec >= min_precision:
            candidates.append((thresh, prec))
    
    if not candidates:
        raise ValueError(
            f"No threshold satisfies target_recall={target_recall} "
            f"and min_precision={min_precision}. Model too weak for policy."
        )
    
    candidates.sort(key=lambda x: (x[1], x[0]), reverse=True)
    
    return float(candidates[0][0])


def risk_bucket(score: float, bucket_cutoffs: list[float]) -> str:
    if len(bucket_cutoffs) != 2:
        raise ValueError("bucket_cutoffs must have exactly 2 values")
    
    low_med, med_high = bucket_cutoffs
    
    if score < low_med:
        return "low"
    elif score < med_high:
        return "med"
    else:
        return "high"


def apply_policy(
    y_proba,
    threshold: float,
    bucket_cutoffs: list[float],
) -> dict:
    y_proba = np.asarray(y_proba)
    
    buckets = [risk_bucket(score, bucket_cutoffs) for score in y_proba]
    
    return {
        "threshold": threshold,
        "buckets": buckets,
    }
