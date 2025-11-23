import pandas as pd
import pytest

from pmrisk.features.tabular import build_tabular_features


def _make_df():
    return pd.DataFrame(
        {
            "engine_id": [1, 1, 1, 1, 1, 1],
            "cycle": [1, 2, 3, 4, 5, 6],
            "sensor_1": [10, 20, 30, 40, 50, 60],
        }
    )


def _make_spec():
    return {
        "windows": [3],
        "stats": ["mean"],
        "lags": [1],
        "deltas": [1],
        "include_slopes": False,
    }


def test_leakage():
    df = _make_df()
    spec = _make_spec()

    f1 = build_tabular_features(df, spec, window_l=1, filter_after=False)

    df_future = df.copy()
    df_future.loc[df_future["cycle"] == 6, "sensor_1"] = 9999
    f2 = build_tabular_features(df_future, spec, window_l=1, filter_after=False)

    feature_cols = [c for c in f1.columns if c not in ["engine_id", "cycle"]]
    assert feature_cols, "No feature columns generated"

    cycle5_f1 = f1.loc[f1["cycle"] == 5, feature_cols].iloc[0]
    cycle5_f2 = f2.loc[f2["cycle"] == 5, feature_cols].iloc[0]

    for col in feature_cols:
        assert cycle5_f1[col] == pytest.approx(cycle5_f2[col]), f"Feature {col} differs at cycle 5"


def test_trailing_rolling():
    df = _make_df()
    spec = _make_spec()

    result = build_tabular_features(df, spec, window_l=1, filter_after=False)

    cycle4_val = result.loc[result["cycle"] == 4, "sensor_1__roll3__mean"].iloc[0]
    assert cycle4_val == pytest.approx(30.0)


def test_lag_delta():
    df = _make_df()
    spec = _make_spec()

    result = build_tabular_features(df, spec, window_l=1, filter_after=False)
    cycle4_row = result.loc[result["cycle"] == 4].iloc[0]

    assert cycle4_row["sensor_1__lag1"] == pytest.approx(30.0)
    assert cycle4_row["sensor_1__delta1"] == pytest.approx(10.0)
