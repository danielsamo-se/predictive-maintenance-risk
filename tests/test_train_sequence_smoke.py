import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pmrisk.models.model_builder import build_sequence_model
from pmrisk.models.sequence_data import build_sequence_pipeline
from pmrisk.models.torch_dataset import SequenceWindowDataset
from pmrisk.models.train_sequence import (
    eval_loss,
    eval_metrics,
    filter_index_by_engine_ids,
    train_one_epoch,
    train_sequence_model,
)


def test_train_sequence_smoke() -> None:
    np.random.seed(42)
    torch.manual_seed(42)

    rows: list[dict] = []
    for engine_id in (1, 2):
        for cycle in range(1, 21):
            label = 1 if cycle >= 18 else 0
            rows.append(
                {
                    "engine_id": engine_id,
                    "cycle": cycle,
                    "op_setting_1": float(np.random.randn()),
                    "sensor_1": float(np.random.randn()),
                    "sensor_2": float(np.random.randn()),
                    "label": label,
                }
            )

    df = pd.DataFrame(rows)
    window_length = 10

    pipeline = build_sequence_pipeline(
        df,
        train_engine_ids=[1],
        window_length=window_length,
        label_col="label",
    )

    feature_columns = pipeline["feature_columns"]
    engine_arrays = pipeline["engine_arrays"]
    index = pipeline["index"]

    train_index = filter_index_by_engine_ids(index, engine_ids={1})
    val_index = filter_index_by_engine_ids(index, engine_ids={2})

    assert len(train_index) > 0
    assert len(val_index) > 0

    train_dataset = SequenceWindowDataset(engine_arrays, train_index, window_length)
    val_dataset = SequenceWindowDataset(engine_arrays, val_index, window_length)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    model = build_sequence_model(
        {
            "model_type": "cnn",
            "n_features": len(feature_columns),
            "window_l": window_length,
            "hidden_channels": 8,
        }
    )

    device = torch.device("cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_loss = train_one_epoch(model, train_loader, optimizer, device)
    val_loss = eval_loss(model, val_loader, device)

    assert isinstance(train_loss, float)
    assert isinstance(val_loss, float)
    assert train_loss >= 0.0
    assert val_loss >= 0.0

    metrics = eval_metrics(model, val_loader, device)

    assert isinstance(metrics["loss"], float)
    assert metrics["loss"] >= 0.0
    assert isinstance(metrics["pr_auc"], float)
    assert 0.0 <= metrics["pr_auc"] <= 1.0


def test_train_sequence_model_with_early_stopping() -> None:
    np.random.seed(42)
    torch.manual_seed(42)

    rows: list[dict] = []
    for engine_id in (1, 2):
        for cycle in range(1, 21):
            label = 1 if cycle >= 18 else 0
            rows.append(
                {
                    "engine_id": engine_id,
                    "cycle": cycle,
                    "op_setting_1": float(np.random.randn()),
                    "sensor_1": float(np.random.randn()),
                    "sensor_2": float(np.random.randn()),
                    "label": label,
                }
            )

    df = pd.DataFrame(rows)
    window_length = 10

    pipeline = build_sequence_pipeline(
        df,
        train_engine_ids=[1],
        window_length=window_length,
        label_col="label",
    )

    feature_columns = pipeline["feature_columns"]
    engine_arrays = pipeline["engine_arrays"]
    index = pipeline["index"]

    train_index = filter_index_by_engine_ids(index, engine_ids={1})
    val_index = filter_index_by_engine_ids(index, engine_ids={2})

    train_dataset = SequenceWindowDataset(engine_arrays, train_index, window_length)
    val_dataset = SequenceWindowDataset(engine_arrays, val_index, window_length)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    model = build_sequence_model(
        {
            "model_type": "cnn",
            "n_features": len(feature_columns),
            "window_l": window_length,
            "hidden_channels": 8,
        }
    )

    device = torch.device("cpu")
    model.to(device)

    result = train_sequence_model(
        model,
        train_loader,
        val_loader,
        device,
        n_epochs=5,
        patience=2,
        lr=1e-3,
    )

    history = result["history"]
    best_state_dict = result["best_state_dict"]
    best_metrics = result["best_metrics"]

    assert len(history) >= 1
    assert isinstance(best_state_dict, dict)
    assert len(best_state_dict) > 0
    assert "val_pr_auc" in best_metrics
    assert isinstance(best_metrics["val_pr_auc"], float)
    assert 0.0 <= best_metrics["val_pr_auc"] <= 1.0

