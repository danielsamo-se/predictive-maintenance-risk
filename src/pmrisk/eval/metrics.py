import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_proba, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    
    if len(y_true) != len(y_proba):
        raise ValueError(f"y_true and y_proba must have same length, got {len(y_true)} and {len(y_proba)}")
    
    if np.any(y_proba < 0) or np.any(y_proba > 1):
        raise ValueError("y_proba must be in [0, 1]")
    
    y_pred = (y_proba >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    return {
        "pr_auc": average_precision_score(y_true, y_proba),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }
