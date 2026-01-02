"""Train model on FD001 data"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from pmrisk.models.model_builder import build_sequence_model
from pmrisk.models.model_versions import save_sequence_model_production
from pmrisk.models.sequence_data import build_sequence_pipeline
from pmrisk.models.torch_dataset import SequenceWindowDataset
from pmrisk.models.train_sequence import (
    compute_binary_metrics_at_threshold,
    eval_metrics,
    filter_index_by_engine_ids,
    predict_logits,
    select_threshold_for_target_recall,
    train_sequence_model,
)
from pmrisk.split.splitter import split_engine_ids


def _make_json_serializable(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(item) for item in obj]
    else:
        return obj


def main() -> None:
    parser = argparse.ArgumentParser(description="Train sequence model on FD001")
    parser.add_argument(
        "--data",
        default="data/processed/fd001_train_labeled.parquet",
        help="Path to labeled parquet",
    )
    parser.add_argument("--model-name", default="seq", help="Model name")
    parser.add_argument("--version", default="v0-smoke", help="Model version")
    parser.add_argument(
        "--model-type",
        default="cnn",
        choices=["cnn", "gru"],
        help="Model type",
    )
    parser.add_argument("--window-l", type=int, default=50, help="Window length")

    parser.add_argument("--max-train-engines", type=int, default=None, help="Cap train engines")
    parser.add_argument("--max-val-engines", type=int, default=None, help="Cap val engines")
    parser.add_argument("--max-test-engines", type=int, default=None, help="Cap test engines")

    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--n-epochs", type=int, default=3, help="Epochs")
    parser.add_argument("--patience", type=int, default=1, help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--target-recall", type=float, default=0.85, help="Target recall on val")

    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data file: {data_path}")

    split_config_path = Path("configs/split.yaml")
    if not split_config_path.exists():
        raise FileNotFoundError(f"Missing split config: {split_config_path}")

    df = pd.read_parquet(data_path)

    for col in ("engine_id", "cycle", "label"):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    with open(split_config_path, "r", encoding="utf-8") as f:
        split_cfg = yaml.safe_load(f)

    all_engine_ids = sorted(df["engine_id"].unique())
    splits = split_engine_ids(
        all_engine_ids,
        train_ratio=split_cfg["train_ratio"],
        val_ratio=split_cfg["val_ratio"],
        test_ratio=split_cfg["test_ratio"],
        seed=split_cfg["seed"],
    )

    train_engine_ids = splits["train"]
    val_engine_ids = splits["val"]
    test_engine_ids = splits["test"]

    if args.max_train_engines is not None:
        train_engine_ids = train_engine_ids[: args.max_train_engines]
    if args.max_val_engines is not None:
        val_engine_ids = val_engine_ids[: args.max_val_engines]
    if args.max_test_engines is not None:
        test_engine_ids = test_engine_ids[: args.max_test_engines]

    if not train_engine_ids:
        raise ValueError("Empty train split")
    if not val_engine_ids:
        raise ValueError("Empty val split")
    if not test_engine_ids:
        raise ValueError("Empty test split")

    pipeline = build_sequence_pipeline(
        df,
        train_engine_ids=train_engine_ids,
        window_length=args.window_l,
        label_col="label",
    )

    feature_columns = pipeline["feature_columns"]
    engine_arrays = pipeline["engine_arrays"]
    index = pipeline["index"]

    train_index = filter_index_by_engine_ids(index, engine_ids=set(train_engine_ids))
    val_index = filter_index_by_engine_ids(index, engine_ids=set(val_engine_ids))
    test_index = filter_index_by_engine_ids(index, engine_ids=set(test_engine_ids))

    train_dataset = SequenceWindowDataset(engine_arrays, train_index, args.window_l)
    val_dataset = SequenceWindowDataset(engine_arrays, val_index, args.window_l)
    test_dataset = SequenceWindowDataset(engine_arrays, test_index, args.window_l)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    hparams = {
        "model_type": args.model_type,
        "n_features": len(feature_columns),
        "window_l": args.window_l,
    }
    if args.model_type == "cnn":
        hparams.update({"hidden_channels": 8, "kernel_size": 3, "dropout_p": 0.0})
    else:
        hparams.update({"hidden_size": 32, "num_layers": 1, "dropout_p": 0.0})

    print(
        f"data={data_path} model={args.model_type} L={args.window_l} F={len(feature_columns)} "
        f"engines(train/val/test)={len(train_engine_ids)}/{len(val_engine_ids)}/{len(test_engine_ids)} "
        f"windows(train/val/test)={len(train_index)}/{len(val_index)}/{len(test_index)}"
    )

    model = build_sequence_model(hparams)
    device = torch.device("cpu")
    model.to(device)

    result = train_sequence_model(
        model,
        train_loader,
        val_loader,
        device,
        n_epochs=args.n_epochs,
        patience=args.patience,
        lr=args.lr,
    )

    best_state_dict = result["best_state_dict"]
    best_metrics = result["best_metrics"]
    history = result["history"]

    model.load_state_dict(best_state_dict)
    model.eval()

    logits_val, y_true_val = predict_logits(model, val_loader, device)
    y_val = y_true_val.view(-1).cpu()
    s_val = torch.sigmoid(logits_val.view(-1).cpu())

    threshold = select_threshold_for_target_recall(y_val, s_val, args.target_recall)
    val_at_thr = compute_binary_metrics_at_threshold(y_val, s_val, threshold)

    test_metrics = eval_metrics(model, test_loader, device, threshold=threshold)

    print(
        f"val: pr_auc={best_metrics['val_pr_auc']:.4f} loss={best_metrics['val_loss']:.4f} "
        f"thr(recall>={args.target_recall:.2f})={threshold:.4f} "
        f"p/r/f1={val_at_thr['precision']:.4f}/{val_at_thr['recall']:.4f}/{val_at_thr['f1']:.4f}"
    )
    print(
        f"test: pr_auc={test_metrics['pr_auc']:.4f} loss={test_metrics['loss']:.4f} "
        f"p/r/f1={test_metrics['precision']:.4f}/{test_metrics['recall']:.4f}/{test_metrics['f1']:.4f}"
    )

    if "scaler_params" not in pipeline:
        raise ValueError("Missing scaler_params in pipeline")
    
    sp = pipeline["scaler_params"]
    required_keys = ["feature_columns", "mean", "std"]
    for key in required_keys:
        if key not in sp:
            raise ValueError(f"Missing key '{key}' in scaler_params")
    
    if len(sp["feature_columns"]) != len(sp["mean"]) or len(sp["feature_columns"]) != len(sp["std"]):
        raise ValueError(
            f"Scaler params length mismatch: "
            f"feature_columns={len(sp['feature_columns'])}, mean={len(sp['mean'])}, std={len(sp['std'])}"
        )
    
    if len(sp["feature_columns"]) != hparams["n_features"]:
        raise ValueError(
            f"Feature count mismatch: scaler has {len(sp['feature_columns'])}, "
            f"hparams has {hparams['n_features']}"
        )

    scaler_params_json = _make_json_serializable(sp)

    metadata = {
        "hparams": hparams,
        "best_metrics": best_metrics,
        "threshold": float(threshold),
        "target_recall": float(args.target_recall),
        "val_metrics_at_threshold": val_at_thr,
        "test_metrics": test_metrics,
        "scaler_params": scaler_params_json,
        "feature_columns": feature_columns,
        "train_engines_count": len(train_engine_ids),
        "val_engines_count": len(val_engine_ids),
        "test_engines_count": len(test_engine_ids),
        "train_windows_count": len(train_index),
        "val_windows_count": len(val_index),
        "test_windows_count": len(test_index),
        "split_config": {
            "train_ratio": split_cfg["train_ratio"],
            "val_ratio": split_cfg["val_ratio"],
            "test_ratio": split_cfg["test_ratio"],
            "seed": split_cfg["seed"],
        },
        "history": history,
    }

    root = Path("models") / "production"
    version_dir = save_sequence_model_production(
        model_name=args.model_name,
        version=args.version,
        state_dict=best_state_dict,
        metadata=metadata,
        root=root,
        set_active=True,
    )

    print(f"saved: {version_dir} (active={args.version})")


if __name__ == "__main__":
    main()
