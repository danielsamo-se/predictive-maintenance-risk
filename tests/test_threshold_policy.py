import pytest

from pmrisk.eval.thresholds import apply_policy, risk_bucket, select_threshold


def test_select_threshold_happy_path():
    y_true = [0, 0, 0, 1, 1]
    y_proba = [0.10, 0.20, 0.30, 0.80, 0.90]
    target_recall = 1.0
    min_precision = 0.50
    
    threshold = select_threshold(y_true, y_proba, target_recall, min_precision)
    
    assert threshold == 0.80


def test_select_threshold_guardrail_fail():
    y_true = [0, 0, 0, 1, 1]
    y_proba = [0.60, 0.60, 0.60, 0.60, 0.60]
    target_recall = 0.85
    min_precision = 0.50
    
    with pytest.raises(ValueError, match="Model too weak for policy"):
        select_threshold(y_true, y_proba, target_recall, min_precision)


def test_risk_bucket():
    bucket_cutoffs = [0.2, 0.5]
    
    assert risk_bucket(0.19, bucket_cutoffs) == "low"
    assert risk_bucket(0.2, bucket_cutoffs) == "med"
    assert risk_bucket(0.49, bucket_cutoffs) == "med"
    assert risk_bucket(0.5, bucket_cutoffs) == "high"


def test_apply_policy():
    y_proba = [0.1, 0.2, 0.7]
    threshold = 0.3
    bucket_cutoffs = [0.2, 0.5]
    
    result = apply_policy(y_proba, threshold, bucket_cutoffs)
    
    assert result["threshold"] == 0.3
    assert result["buckets"] == ["low", "med", "high"]
