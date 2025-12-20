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


def test_window_alignment_values():
    df = pd.DataFrame(
        {
            "engine_id": [1, 1, 1, 1, 1],
            "cycle": [1, 2, 3, 4, 5],
            "sensor_1": [10, 20, 30, 40, 50],
            "label": [0, 0, 0, 0, 0],
        }
    )

    feature_columns = get_sequence_feature_columns(df)
    engine_arrays = make_engine_arrays(df, feature_columns, label_col="label")

    L = 3
    index = make_sequence_index(engine_arrays, L)

    entry = next(e for e in index if e["pred_cycle"] == 5)
    window = extract_window(engine_arrays, entry["engine_id"], entry["end_pos"], L)

    expected = np.array([[30], [40], [50]], dtype=np.float32)
    assert window.shape == (3, 1)
    assert np.allclose(window, expected)


def test_scaler_fit_train_only_and_no_nan():
    df = pd.DataFrame(
        {
            "engine_id": [1, 1, 2, 2],
            "cycle": [1, 2, 1, 2],
            "sensor_1": [0.0, 2.0, 100.0, 200.0],
            "label": [0, 0, 0, 0],
        }
    )

    feature_columns = get_sequence_feature_columns(df)

    params = fit_standard_scaler_params(df, feature_columns, train_engine_ids=[1])
    df_scaled = apply_standard_scaler(df, params)

    train_scaled = df_scaled[df_scaled["engine_id"] == 1]["sensor_1"].to_numpy()
    assert np.allclose(train_scaled, np.array([-1.0, 1.0], dtype=float))

    x = df_scaled[feature_columns].to_numpy(dtype=float)
    assert np.isfinite(x).all()


def test_constant_feature_std_guard():
    df = pd.DataFrame(
        {
            "engine_id": [1, 1, 1],
            "cycle": [1, 2, 3],
            "sensor_1": [5.0, 5.0, 5.0],
            "label": [0, 0, 0],
        }
    )

    feature_columns = get_sequence_feature_columns(df)
    params = fit_standard_scaler_params(df, feature_columns, train_engine_ids=[1])
    df_scaled = apply_standard_scaler(df, params)

    x = df_scaled[feature_columns].to_numpy(dtype=float)
    assert np.isfinite(x).all()
    assert np.allclose(df_scaled["sensor_1"].to_numpy(dtype=float), 0.0)