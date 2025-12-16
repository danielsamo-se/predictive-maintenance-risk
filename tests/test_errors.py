import pandas as pd
import pytest

from pmrisk.eval.errors import summarize_errors


def test_summarize_errors_missing_cols():
    df = pd.DataFrame({"engine_id": [1], "cycle": [10], "label": [0]})
    
    with pytest.raises(ValueError, match="Missing required columns"):
        summarize_errors(df, horizon_n=30, early_alarm_slack=10)


def test_summarize_errors_fp_fn_counts_and_order():
    df = pd.DataFrame({
        "engine_id": [1, 1, 1, 2, 2],
        "cycle": [10, 20, 30, 15, 25],
        "label": [0, 1, 0, 0, 1],
        "y_proba": [0.8, 0.2, 0.9, 0.7, 0.3],
        "y_pred": [1, 0, 1, 1, 0],
    })

    result = summarize_errors(df, horizon_n=30, early_alarm_slack=10, top_k=20)

    fps = result["false_positives"]
    fns = result["false_negatives"]

    assert len(fps) == 3
    assert len(fns) == 2

    assert [fp["y_proba"] for fp in fps] == sorted([fp["y_proba"] for fp in fps], reverse=True)

    assert [fn["y_proba"] for fn in fns] == sorted([fn["y_proba"] for fn in fns], reverse=True)

    for item in fps + fns:
        for key in ["engine_id", "cycle", "y_proba", "label", "y_pred"]:
            assert key in item


def test_summarize_errors_fp_kind_with_remaining():
    horizon_n = 30
    early_alarm_slack = 10
    
    df = pd.DataFrame({
        "engine_id": [1, 1, 1],
        "cycle": [100, 110, 120],
        "label": [0, 0, 0],
        "y_proba": [0.9, 0.8, 0.95],
        "y_pred": [1, 1, 1],
        "remaining": [35, 50, 0],
    })
    
    result = summarize_errors(df, horizon_n, early_alarm_slack, top_k=20)
    fps = result["false_positives"]
        
    fp_rem_35 = next(f for f in fps if f["remaining"] == 35)
    fp_rem_50 = next(f for f in fps if f["remaining"] == 50)
    fp_rem_0  = next(f for f in fps if f["remaining"] == 0)
    
    assert fp_rem_35["fp_kind"] == "premature_alarm"
    assert fp_rem_50["fp_kind"] == "noise_fp"
    assert fp_rem_0["fp_kind"] == "at_or_past_failure"
