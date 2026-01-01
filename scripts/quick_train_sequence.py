"""Train model on FD001 data"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pmrisk.models.model_builder import build_sequence_model
from pmrisk.models.model_versions import save_sequence_model_production
from pmrisk.models.sequence_data import build_sequence_pipeline
from pmrisk.models.torch_dataset import SequenceWindowDataset
from pmrisk.models.train_sequence import (
    filter_index_by_engine_ids,
    predict_logits,
    select_threshold_for_target_recall,
    compute_binary_metrics_at_threshold,
    train_sequence_model,
)


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
        help="Model type: cnn or gru",
    )
    parser.add_argument("--window-l", type=int, default=50, help="Window length")
    parser.add_argument("--max-train-engines", type=int, default=50, help="Max train engines")
    parser.add_argument("--max-val-engines", type=int, default=10, help="Max val engines")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--n-epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=1, help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--target-recall", type=float, default=0.85, help="Target recall for threshold selection"
    )
    
    args = parser.parse_args()
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    required_cols = ["engine_id", "cycle", "label"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    all_engine_ids = sorted(df["engine_id"].unique())
    train_engine_ids = all_engine_ids[: args.max_train_engines]
    val_engine_ids = all_engine_ids[
        args.max_train_engines : args.max_train_engines + args.max_val_engines
    ]
    
    if len(train_engine_ids) == 0:
        raise ValueError("No train engines available")
    if len(val_engine_ids) == 0:
        raise ValueError("No val engines available")
    
    print(f"Train engines: {len(train_engine_ids)}")
    print(f"Val engines: {len(val_engine_ids)}")
    
    print(f"Building sequence pipeline (window_l={args.window_l})")
    pipeline = build_sequence_pipeline(
        df,
        train_engine_ids=train_engine_ids,
        window_length=args.window_l,
        label_col="label",
    )
    
    feature_columns = pipeline["feature_columns"]
    engine_arrays = pipeline["engine_arrays"]
    index = pipeline["index"]
    
    print(f"Feature columns: {len(feature_columns)}")
    print(f"Total windows: {len(index)}")
    
    train_index = filter_index_by_engine_ids(index, engine_ids=set(train_engine_ids))
    val_index = filter_index_by_engine_ids(index, engine_ids=set(val_engine_ids))
    
    print(f"Train windows: {len(train_index)}")
    print(f"Val windows: {len(val_index)}")
    
    train_dataset = SequenceWindowDataset(engine_arrays, train_index, args.window_l)
    val_dataset = SequenceWindowDataset(engine_arrays, val_index, args.window_l)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    hparams = {
        "model_type": args.model_type,
        "n_features": len(feature_columns),
        "window_l": args.window_l,
    }
    
    if args.model_type == "cnn":
        hparams.update({
            "hidden_channels": 8,
            "kernel_size": 3,
            "dropout_p": 0.0,
        })
    elif args.model_type == "gru":
        hparams.update({
            "hidden_size": 32,
            "num_layers": 1,
            "dropout_p": 0.0,
        })
    
    print(f"\nBuilding model: {args.model_type.upper()}")
    print(f"  n_features: {hparams['n_features']}")
    print(f"  window_l: {hparams['window_l']}")
    if args.model_type == "cnn":
        print(f"  hidden_channels: {hparams['hidden_channels']}")
        print(f"  kernel_size: {hparams['kernel_size']}")
    elif args.model_type == "gru":
        print(f"  hidden_size: {hparams['hidden_size']}")
        print(f"  num_layers: {hparams['num_layers']}")
    print(f"  dropout_p: {hparams['dropout_p']}")
    
    model = build_sequence_model(hparams)
    
    device = torch.device("cpu")
    model.to(device)
    
    print(f"\nTraining for {args.n_epochs} epochs (patience={args.patience})...")
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
    
    print(f"\nTraining completed after {len(history)} epochs")
    print(f"Best val_pr_auc: {best_metrics['val_pr_auc']:.4f}")
    print(f"Best val_loss: {best_metrics['val_loss']:.4f}")
    
    model.load_state_dict(best_state_dict)
    model.eval()
    
    print(f"\nSelecting threshold for target_recall={args.target_recall}...")
    logits, y_true = predict_logits(model, val_loader, device)
    y_true_1d = y_true.view(-1).cpu()
    scores = torch.sigmoid(logits.view(-1).cpu())
    
    threshold = select_threshold_for_target_recall(y_true_1d, scores, args.target_recall)
    val_metrics_at_threshold = compute_binary_metrics_at_threshold(y_true_1d, scores, threshold)
    
    print(f"Selected threshold: {threshold:.4f}")
    print(f"Val precision @ threshold: {val_metrics_at_threshold['precision']:.4f}")
    print(f"Val recall @ threshold: {val_metrics_at_threshold['recall']:.4f}")
    print(f"Val f1 @ threshold: {val_metrics_at_threshold['f1']:.4f}")
    
    metadata = {
        "hparams": hparams,
        "best_metrics": best_metrics,
        "threshold": threshold,
        "target_recall": args.target_recall,
        "val_metrics_at_threshold": val_metrics_at_threshold,
        "train_engines_count": len(train_engine_ids),
        "val_engines_count": len(val_engine_ids),
        "train_windows_count": len(train_index),
        "val_windows_count": len(val_index),
    }
    
    root = Path("models") / "production"
    print(f"\nSaving model to {root / args.model_name / args.version}...")
    
    version_dir = save_sequence_model_production(
        model_name=args.model_name,
        version=args.version,
        state_dict=best_state_dict,
        metadata=metadata,
        root=root,
        set_active=True,
    )
    
    print(f"Model saved: {version_dir}")
    print(f"Active version set: {args.version}")
    print("\n=== Best Metrics ===")
    print(f"  val_pr_auc: {best_metrics['val_pr_auc']:.4f}")
    print(f"  val_loss: {best_metrics['val_loss']:.4f}")
    print(f"\n=== Threshold Policy ===")
    print(f"  threshold: {threshold:.4f}")
    print(f"  target_recall: {args.target_recall:.4f}")
    print(f"  precision: {val_metrics_at_threshold['precision']:.4f}")
    print(f"  recall: {val_metrics_at_threshold['recall']:.4f}")
    print(f"  f1: {val_metrics_at_threshold['f1']:.4f}")


if __name__ == "__main__":
    main()
