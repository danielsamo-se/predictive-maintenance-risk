import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from pmrisk.features.sequence import get_sequence_feature_columns, make_engine_arrays, make_sequence_index
from pmrisk.models.torch_dataset import SequenceWindowDataset


def test_dataset_alignment_and_values():
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

    l = 3
    index = make_sequence_index(engine_arrays, l)
    ds = SequenceWindowDataset(engine_arrays, index, l)

    entry_idx = next(i for i, e in enumerate(index) if e["pred_cycle"] == 5)
    x_win, y = ds[entry_idx]

    assert x_win.shape == (3, 1)
    assert x_win.dtype == torch.float32

    expected = np.array([[30], [40], [50]], dtype=np.float32)
    assert np.allclose(x_win.numpy(), expected)

    assert y.dtype == torch.int64
    assert y.item() == 0


def test_dataset_raises_on_bad_index_out_of_range():
    df = pd.DataFrame(
        {
            "engine_id": [1, 1, 1],
            "cycle": [1, 2, 3],
            "sensor_1": [10, 20, 30],
            "label": [0, 0, 0],
        }
    )

    feature_columns = get_sequence_feature_columns(df)
    engine_arrays = make_engine_arrays(df, feature_columns, label_col="label")

    bad_index = [{"engine_id": 1, "pred_cycle": 99, "end_pos": 99}]
    with pytest.raises(ValueError, match="out of range"):
        SequenceWindowDataset(engine_arrays, bad_index, window_length=3)


def test_dataset_raises_when_end_pos_before_window_start():
    df = pd.DataFrame(
        {
            "engine_id": [1, 1, 1],
            "cycle": [1, 2, 3],
            "sensor_1": [10, 20, 30],
            "label": [0, 0, 0],
        }
    )

    feature_columns = get_sequence_feature_columns(df)
    engine_arrays = make_engine_arrays(df, feature_columns, label_col="label")

    bad_index = [{"engine_id": 1, "pred_cycle": 2, "end_pos": 1}]
    with pytest.raises(ValueError, match="window_length - 1"):
        SequenceWindowDataset(engine_arrays, bad_index, window_length=3)
