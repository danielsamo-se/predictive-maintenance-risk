import numpy as np
import pandas as pd

from pmrisk.features.sequence import extract_window
from pmrisk.models.sequence_data import build_sequence_pipeline


def test_build_sequence_pipeline_end_to_end_no_nan():
    df = pd.DataFrame({
        "engine_id": [1, 1, 1, 1, 2, 2, 2, 2],
        "cycle": [1, 2, 3, 4, 1, 2, 3, 4],
        "sensor_1": [0, 1, 2, 3, 100, 101, 102, 103],
        "label": [0, 0, 0, 1, 0, 0, 1, 1],
    })
    
    train_engine_ids = [1]
    window_length = 3
    
    result = build_sequence_pipeline(df, train_engine_ids, window_length, label_col="label")
    
    assert result["feature_columns"] == ["sensor_1"]
    
    index = result["index"]
    assert len(index) == 4
    
    engine_arrays = result["engine_arrays"]
    
    for engine_id in [1, 2]:
        X = engine_arrays[engine_id]["X"]
        assert X.dtype == np.float32
        assert np.isfinite(X).all()

    x_train = engine_arrays[1]["X"][:, 0].astype(float)
    assert abs(float(x_train.mean())) < 1e-6
    
    last_entry = index[-1]
    sample_window = extract_window(
        engine_arrays,
        last_entry["engine_id"],
        last_entry["end_pos"],
        window_length,
    )
    assert np.isfinite(sample_window).all()
    
    scaler_params = result["scaler_params"]
    mean = scaler_params["mean"][0]
