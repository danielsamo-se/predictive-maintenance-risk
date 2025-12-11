import pytest

from pmrisk.eval.calibration import compute_calibration


def test_compute_calibration_shapes_and_keys():
    y_true = [0, 0, 1, 1, 0, 1, 0, 1, 0, 1]
    y_proba = [0.1, 0.2, 0.7, 0.8, 0.3, 0.9, 0.2, 0.8, 0.1, 0.9]

    result = compute_calibration(y_true, y_proba, n_bins=5, strategy="uniform")

    assert "brier_score" in result
    assert "bins" in result
    assert isinstance(result["bins"], list)
    assert len(result["bins"]) > 0

    for b in result["bins"]:
        assert "bin_idx" in b
        assert "count" in b
        assert "mean_pred" in b
        assert "frac_pos" in b


def test_compute_calibration_brier_range():
    y_true = [0, 0, 1, 1]
    y_proba = [0.1, 0.2, 0.7, 0.8]

    result = compute_calibration(y_true, y_proba)

    assert 0.0 <= result["brier_score"] <= 1.0


def test_compute_calibration_invalid_inputs():
    y_true = [0, 1, 0, 1]
    y_proba_valid = [0.2, 0.8, 0.3, 0.7]

    with pytest.raises(ValueError, match="same length"):
        compute_calibration(y_true, y_proba_valid[:-1])

    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        compute_calibration(y_true, [0.2, 1.5, 0.3, 0.7])

    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        compute_calibration(y_true, [-0.1, 0.8, 0.3, 0.7])

    with pytest.raises(ValueError, match="strategy must be"):
        compute_calibration(y_true, y_proba_valid, strategy="invalid")

    with pytest.raises(ValueError, match="n_bins must be"):
        compute_calibration(y_true, y_proba_valid, n_bins=1)


def test_compute_calibration_counts_cover_all_samples():
    y_true = [0, 1, 0, 1, 0, 1]
    y_proba = [0.05, 0.06, 0.07, 0.95, 0.96, 0.97]

    out = compute_calibration(y_true, y_proba, n_bins=10, strategy="uniform")
    assert sum(b["count"] for b in out["bins"]) == len(y_true)

