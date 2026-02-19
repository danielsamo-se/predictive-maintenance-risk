"""
Train all models and compare metrics (val + test)
Select best by PR-AUC on validation
"""

import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from pmrisk.anomaly.isoforest import fit_isoforest, score_isoforest
from pmrisk.eval.calibration import compute_calibration
from pmrisk.eval.metrics import compute_metrics
from pmrisk.eval.thresholds import select_threshold
from pmrisk.features.contracts import validate_tabular_features
from pmrisk.features.tabular import build_tabular_features
from pmrisk.models.model_builder import build_sequence_model
from pmrisk.models.sequence_data import build_sequence_pipeline
from pmrisk.models.torch_dataset import SequenceWindowDataset
from pmrisk.models.train_sequence import (
    filter_index_by_engine_ids,
    predict_logits,
    select_threshold_for_policy,
    train_sequence_model,
)
from pmrisk.split.splitter import split_engine_ids


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def topk_stats(labels, scores, k):
    n = len(labels)
    if n == 0:
        return {"precision": float("nan"), "lift": float("nan"), "capture": float("nan")}
    m = max(1, int(math.ceil(k * n)))
    order = np.argsort(-scores)
    top_labels = labels[order[:m]]
    pos = int(labels.sum())
    base_rate = float(labels.mean())
    top_pos = int(top_labels.sum())
    precision = float(top_pos / m)
    lift = float(precision / base_rate) if base_rate > 0 else float("nan")
    capture = float(top_pos / pos) if pos > 0 else float("nan")
    return {"precision": precision, "lift": lift, "capture": capture}


def train_sequence(model_type, df_labeled, splits, window_l, thresh_cfg, n_epochs, patience):
    pipeline = build_sequence_pipeline(
        df_labeled,
        train_engine_ids=splits["train"],
        window_length=window_l,
        label_col="label",
    )

    engine_arrays = pipeline["engine_arrays"]
    index = pipeline["index"]

    loaders = {}
    for split_name in ("train", "val", "test"):
        idx = filter_index_by_engine_ids(index, set(splits[split_name]))
        ds = SequenceWindowDataset(engine_arrays, idx, window_l)
        loaders[split_name] = DataLoader(ds, batch_size=32, shuffle=False)

    hparams = {
        "model_type": model_type,
        "n_features": len(pipeline["feature_columns"]),
        "window_l": window_l,
    }
    if model_type == "cnn":
        hparams.update({"hidden_channels": 8, "kernel_size": 3, "dropout_p": 0.0})
    else:
        hparams.update({"hidden_size": 32, "num_layers": 1, "dropout_p": 0.0})

    model = build_sequence_model(hparams)
    device = torch.device("cpu")
    model.to(device)

    result = train_sequence_model(
        model, loaders["train"], loaders["val"], device,
        n_epochs=n_epochs, patience=patience, lr=1e-3,
    )

    model.load_state_dict(result["best_state_dict"])
    model.eval()

    logits_val, y_true_val = predict_logits(model, loaders["val"], device)
    y_val = y_true_val.view(-1)
    s_val = torch.sigmoid(logits_val.view(-1))

    threshold = select_threshold_for_policy(
        y_val, s_val,
        target_recall=thresh_cfg["target_recall"],
        min_precision=thresh_cfg["min_precision"],
    )

    metrics = {}
    for split_name in ("train", "val", "test"):
        logits, y_true = predict_logits(model, loaders[split_name], device)
        y_np = y_true.view(-1).numpy()
        s_np = torch.sigmoid(logits.view(-1)).numpy()
        m = compute_metrics(y_np, s_np, float(threshold))
        cal = compute_calibration(y_np, s_np)
        m["brier"] = cal["brier_score"]
        metrics[split_name] = m

    return {
        "threshold": float(threshold),
        "metrics": metrics,
        "epochs_trained": len(result["history"]),
    }


def main():
    base_cfg = load_config("configs/base.yaml")
    feat_cfg = load_config("configs/features.yaml")
    split_cfg = load_config("configs/split.yaml")
    thresh_cfg = load_config("configs/thresholds.yaml")
    anomaly_cfg = load_config("configs/anomaly.yaml")

    window_l = base_cfg["window_l"]
    random_seed = base_cfg["random_seed"]

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    processed_dir = Path(base_cfg["data_processed_dir"])

    df_full = pd.read_parquet(processed_dir / "fd001_train.parquet")
    df_labels = pd.read_parquet(processed_dir / "fd001_train_labeled.parquet")

    all_engine_ids = sorted(df_full["engine_id"].unique().tolist())
    splits = split_engine_ids(
        all_engine_ids,
        split_cfg["train_ratio"],
        split_cfg["val_ratio"],
        split_cfg["test_ratio"],
        split_cfg["seed"],
    )

    feat_spec = {
        "windows": feat_cfg["windows"],
        "stats": feat_cfg["stats"],
        "lags": feat_cfg["lags"],
        "deltas": feat_cfg["deltas"],
        "include_slopes": feat_cfg["include_slopes"],
    }

    df_feat = build_tabular_features(df_full, feat_spec, window_l, filter_after=True)
    validate_tabular_features(df_feat, window_l)

    label_cols = ["engine_id", "cycle", "label"]
    if "remaining" in df_labels.columns:
        label_cols.append("remaining")

    df_feat = df_feat.merge(df_labels[label_cols], on=["engine_id", "cycle"], how="inner")

    exclude = {"engine_id", "cycle", "label", "remaining", "failure_cycle"}
    feature_columns = sorted([c for c in df_feat.columns if c not in exclude])

    train_df = df_feat[df_feat["engine_id"].isin(splits["train"])]
    val_df = df_feat[df_feat["engine_id"].isin(splits["val"])]
    test_df = df_feat[df_feat["engine_id"].isin(splits["test"])]

    X_train, y_train = train_df[feature_columns], train_df["label"]
    X_val, y_val = val_df[feature_columns], val_df["label"]
    X_test, y_test = test_df[feature_columns], test_df["label"]

    tabular_models = {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=random_seed,
            )),
        ]),
        "lightgbm": lgb.LGBMClassifier(
            random_state=random_seed,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            class_weight="balanced",
            verbosity=-1,
        ),
    }

    results = {}

    for name, model in tabular_models.items():
        print(f"Training {name}")
        model.fit(X_train, y_train)

        proba_val = model.predict_proba(X_val)[:, 1]
        threshold = select_threshold(
            y_val, proba_val,
            thresh_cfg["target_recall"],
            thresh_cfg["min_precision"],
        )

        metrics = {}
        for split_name, (X, y) in [("train", (X_train, y_train)),
                                    ("val", (X_val, y_val)),
                                    ("test", (X_test, y_test))]:
            proba = model.predict_proba(X)[:, 1]
            m = compute_metrics(y, proba, threshold)
            cal = compute_calibration(y, proba)
            m["brier"] = cal["brier_score"]
            metrics[split_name] = m

        results[name] = {"threshold": threshold, "metrics": metrics}
        print(f"  val PR-AUC: {metrics['val']['pr_auc']:.4f}  test PR-AUC: {metrics['test']['pr_auc']:.4f}")

    n_epochs = 20
    patience = 5

    for model_type in ("cnn", "gru"):
        print(f"Training {model_type} (epochs={n_epochs}, patience={patience})")
        r = train_sequence(model_type, df_labels, splits, window_l, thresh_cfg, n_epochs, patience)
        results[model_type] = {"threshold": r["threshold"], "metrics": r["metrics"]}
        print(f"  val PR-AUC: {r['metrics']['val']['pr_auc']:.4f}"
              f"  test PR-AUC: {r['metrics']['test']['pr_auc']:.4f}"
              f"  (stopped at epoch {r['epochs_trained']})")

    print("Training isolation_forest")
    X_train_np = X_train.to_numpy(dtype=np.float64)
    X_val_np = X_val.to_numpy(dtype=np.float64)
    X_test_np = X_test.to_numpy(dtype=np.float64)

    iso_model = fit_isoforest(X_train_np, float(anomaly_cfg["contamination"]), int(anomaly_cfg["random_seed"]))
    val_scores = score_isoforest(iso_model, X_val_np)
    test_scores = score_isoforest(iso_model, X_test_np)

    iso_results = {
        "val_top1": topk_stats(y_val.to_numpy(), val_scores, 0.01),
        "val_top5": topk_stats(y_val.to_numpy(), val_scores, 0.05),
        "test_top1": topk_stats(y_test.to_numpy(), test_scores, 0.01),
        "test_top5": topk_stats(y_test.to_numpy(), test_scores, 0.05),
    }
    print(f"  val top-5% precision: {iso_results['val_top5']['precision']:.4f}")

    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    model_display = {
        "logreg": "Logistic Regression",
        "lightgbm": "LightGBM",
        "cnn": "1D-CNN (PyTorch)",
        "gru": "GRU (PyTorch)",
    }

    lines = []
    lines.append("Model Comparison")
    lines.append("")
    lines.append(f"Split: engine-based {split_cfg['train_ratio']}/{split_cfg['val_ratio']}/{split_cfg['test_ratio']}")
    lines.append(f"Threshold policy: recall >= {thresh_cfg['target_recall']}, precision >= {thresh_cfg['min_precision']}")
    lines.append("")

    for key in ("logreg", "lightgbm", "cnn", "gru"):
        r = results[key]
        lines.append(f"{model_display[key]}")
        lines.append(f"- threshold: {r['threshold']:.4f}")

        for split_name in ("train", "val", "test"):
            m = r["metrics"][split_name]
            cm = m["confusion_matrix"]
            lines.append(f"  {split_name.capitalize()}:")
            lines.append(f"  - PR-AUC: {m['pr_auc']:.4f}")
            lines.append(f"  - ROC-AUC: {m['roc_auc']:.4f}")
            lines.append(f"  - Precision: {m['precision']:.4f}")
            lines.append(f"  - Recall: {m['recall']:.4f}")
            lines.append(f"  - F1: {m['f1']:.4f}")
            lines.append(f"  - Brier: {m['brier']:.4f}")
            lines.append(f"  - CM: tn={cm['tn']}, fp={cm['fp']}, fn={cm['fn']}, tp={cm['tp']}")

        lines.append("")

    lines.append("Overfitting check (train vs val PR-AUC)")
    lines.append("")
    for key in ("logreg", "lightgbm", "cnn", "gru"):
        train_auc = results[key]["metrics"]["train"]["pr_auc"]
        val_auc = results[key]["metrics"]["val"]["pr_auc"]
        gap = train_auc - val_auc
        lines.append(f"- {model_display[key]}: train={train_auc:.4f}, val={val_auc:.4f}, gap={gap:.4f}")
    lines.append("")

    best_key = max(results, key=lambda k: results[k]["metrics"]["val"]["pr_auc"])
    lines.append(f"Selected model: {model_display[best_key]} (highest val PR-AUC)")
    lines.append("")

    lines.append("IsolationForest (unsupervised baseline)")
    lines.append("")
    for split_name in ("val", "test"):
        top1 = iso_results[f"{split_name}_top1"]
        top5 = iso_results[f"{split_name}_top5"]
        lines.append(f"  {split_name.capitalize()}:")
        lines.append(f"  - top_1pct_precision: {top1['precision']:.4f}")
        lines.append(f"  - top_1pct_lift: {top1['lift']:.4f}")
        lines.append(f"  - top_1pct_capture: {top1['capture']:.4f}")
        lines.append(f"  - top_5pct_precision: {top5['precision']:.4f}")
        lines.append(f"  - top_5pct_lift: {top5['lift']:.4f}")
        lines.append(f"  - top_5pct_capture: {top5['capture']:.4f}")
    lines.append("")

    report_path = report_dir / "model_comparison.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    json_path = report_dir / "model_comparison.json"
    raw = {"supervised": {}, "anomaly": iso_results}
    for key in ("logreg", "lightgbm", "cnn", "gru"):
        raw["supervised"][key] = {
            "threshold": results[key]["threshold"],
            "val": results[key]["metrics"]["val"],
            "test": results[key]["metrics"]["test"],
            "train_pr_auc": results[key]["metrics"]["train"]["pr_auc"],
        }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2)

    print(f"Report: {report_path.resolve()}")
    print(f"JSON: {json_path.resolve()}")


if __name__ == "__main__":
    main()