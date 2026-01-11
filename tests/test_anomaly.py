import numpy as np

from pmrisk.anomaly.isoforest import fit_isoforest, score_isoforest


def test_isoforest_scores_outliers_higher():
    rng = np.random.default_rng(42)

    X_train = rng.normal(0.0, 1.0, size=(1000, 5))
    model = fit_isoforest(X_train, contamination=0.02, seed=42)

    X_normal = rng.normal(0.0, 1.0, size=(200, 5))
    X_outliers = rng.normal(8.0, 1.0, size=(20, 5))
    X = np.vstack([X_normal, X_outliers])

    scores = score_isoforest(model, X)

    assert scores.shape == (220,)
    assert np.isfinite(scores).all()
    assert scores[-20:].mean() > scores[:200].mean()
