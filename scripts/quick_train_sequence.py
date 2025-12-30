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
    parser.add_argument("--model-type", default="cnn", help="Model type")
    parser.add_argument("--window-l", type=int, default=50, help="Window length")
    parser.add_argument("--max-train-engines", type=int, default=50, help="Max train engines")
    parser.add_argument("--max-val-engines", type=int, default=10, help="Max val engines")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--n-epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=1, help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    
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
        "hidden_channels": 8,
        "kernel_size": 3,
        "dropout_p": 0.0,
    }
    
    print(f"Building model: {args.model_type}")
    print(f"  n_features: {hparams['n_features']}")
    print(f"  window_l: {hparams['window_l']}")
    
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
    
    metadata = {
        "hparams": hparams,
        "best_metrics": best_metrics,
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
    for k, v in best_metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
