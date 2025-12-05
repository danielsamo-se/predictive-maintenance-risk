"""
Split by engine_id to avoid leakage across the same asset lifecycle
Train two baselines (logreg + lightgbm) and select by PR-AUC on validation
Threshold is chosen by a policy (target recall + minimum precision guardrail)
Report includes model selection, threshold, and metrics
"""

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pmrisk.eval.metrics import compute_metrics
from pmrisk.eval.thresholds import select_threshold
from pmrisk.features.contracts import validate_tabular_features
from pmrisk.features.tabular import build_tabular_features
from pmrisk.split.splitter import split_engine_ids


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    base_cfg = load_config("configs/base.yaml")
    feat_cfg = load_config("configs/features.yaml")
    split_cfg = load_config("configs/split.yaml")
    thresh_cfg = load_config("configs/thresholds.yaml")
    
    horizon_n = base_cfg["horizon_n"]
    window_l = base_cfg["window_l"]
    random_seed = base_cfg["random_seed"]
    featureset_version = feat_cfg["featureset_version"]
    
    processed_dir = Path(base_cfg["data_processed_dir"])

    df_full = pd.read_parquet(processed_dir / "fd001_train.parquet")

    df_labels = pd.read_parquet(processed_dir / "fd001_train_labeled.parquet")[["engine_id", "cycle", "label"]]

    feat_spec = {
        "windows": feat_cfg["windows"],
        "stats": feat_cfg["stats"],
        "lags": feat_cfg["lags"],
        "deltas": feat_cfg["deltas"],
        "include_slopes": feat_cfg["include_slopes"],
    }

    df_feat = build_tabular_features(df_full, feat_spec, window_l, filter_after=True)
    validate_tabular_features(df_feat, window_l)

    df_feat = df_feat.merge(
        df_labels,
        on=["engine_id", "cycle"],
        how="inner",
    )
    
    engine_ids = df_feat["engine_id"].unique().tolist()
    splits = split_engine_ids(
        engine_ids,
        split_cfg["train_ratio"],
        split_cfg["val_ratio"],
        split_cfg["test_ratio"],
        split_cfg["seed"],
    )
    
    train_df = df_feat[df_feat["engine_id"].isin(splits["train"])].copy()
    val_df = df_feat[df_feat["engine_id"].isin(splits["val"])].copy()
    test_df = df_feat[df_feat["engine_id"].isin(splits["test"])].copy()
    
    feature_columns = sorted([c for c in df_feat.columns if c not in ["engine_id", "cycle", "label"]])
    
    X_train = train_df[feature_columns]
    X_val = val_df[feature_columns]
    X_test = test_df[feature_columns]

    y_train = train_df["label"]
    y_val = val_df["label"]
    y_test = test_df["label"]
    
    model_a = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_seed))
    ])
    
    model_b = lgb.LGBMClassifier(
        random_state=random_seed,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        class_weight="balanced",
        verbosity=-1,
    )
    
    model_a.fit(X_train, y_train)
    model_b.fit(X_train, y_train)
    
    proba_val_a = model_a.predict_proba(X_val)[:, 1]
    proba_val_b = model_b.predict_proba(X_val)[:, 1]
    
    metrics_a = compute_metrics(y_val, proba_val_a, threshold=0.5)
    metrics_b = compute_metrics(y_val, proba_val_b, threshold=0.5)
    
    if metrics_a["pr_auc"] >= metrics_b["pr_auc"]:
        selected_model = model_a
        selected_model_type = "logreg"
        proba_val = proba_val_a
        proba_train = model_a.predict_proba(X_train)[:, 1]
        proba_test = model_a.predict_proba(X_test)[:, 1]
    else:
        selected_model = model_b
        selected_model_type = "lightgbm"
        proba_val = proba_val_b
        proba_train = model_b.predict_proba(X_train)[:, 1]
        proba_test = model_b.predict_proba(X_test)[:, 1]
    
    threshold = select_threshold(
        y_val,
        proba_val,
        thresh_cfg["target_recall"],
        thresh_cfg["min_precision"],
    )
    
    train_metrics = compute_metrics(y_train, proba_train, threshold)
    val_metrics = compute_metrics(y_val, proba_val, threshold)
    test_metrics = compute_metrics(y_test, proba_test, threshold)
    
    output_dir = Path(base_cfg["models_production_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_filename = "model.joblib"
    joblib.dump(selected_model, output_dir / model_filename)
    
    metadata = {
        "model_type": selected_model_type,
        "model_filename": model_filename,
        "featureset_version": featureset_version,
        "horizon_n": horizon_n,
        "window_l": window_l,
        "feature_columns": feature_columns,
        "threshold": float(threshold),
        "bucket_cutoffs": thresh_cfg["bucket_cutoffs"],
        "target_recall": thresh_cfg["target_recall"],
        "min_precision": thresh_cfg["min_precision"],
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "baseline_eval.md"
    
    report_lines = [
        "Baseline Model Evaluation",
        "",
        f"Selected model: {selected_model_type}",
        f"Threshold: {threshold:.4f}",
        "",
        "Metrics",
        "",
        "Train:",
        f"- PR-AUC: {train_metrics['pr_auc']:.4f}",
        f"- ROC-AUC: {train_metrics['roc_auc']:.4f}",
        f"- Precision: {train_metrics['precision']:.4f}",
        f"- Recall: {train_metrics['recall']:.4f}",
        f"- F1: {train_metrics['f1']:.4f}",
        "",
        "Val:",
        f"- PR-AUC: {val_metrics['pr_auc']:.4f}",
        f"- ROC-AUC: {val_metrics['roc_auc']:.4f}",
        f"- Precision: {val_metrics['precision']:.4f}",
        f"- Recall: {val_metrics['recall']:.4f}",
        f"- F1: {val_metrics['f1']:.4f}",
        "",
        "Test:",
        f"- PR-AUC: {test_metrics['pr_auc']:.4f}",
        f"- ROC-AUC: {test_metrics['roc_auc']:.4f}",
        f"- Precision: {test_metrics['precision']:.4f}",
        f"- Recall: {test_metrics['recall']:.4f}",
        f"- F1: {test_metrics['f1']:.4f}",
        "",
        "Confusion Matrix (Train):",
        f"- TN: {train_metrics['confusion_matrix']['tn']}",
        f"- FP: {train_metrics['confusion_matrix']['fp']}",
        f"- FN: {train_metrics['confusion_matrix']['fn']}",
        f"- TP: {train_metrics['confusion_matrix']['tp']}",
        "",
        "Confusion Matrix (Val):",
        f"- TN: {val_metrics['confusion_matrix']['tn']}",
        f"- FP: {val_metrics['confusion_matrix']['fp']}",
        f"- FN: {val_metrics['confusion_matrix']['fn']}",
        f"- TP: {val_metrics['confusion_matrix']['tp']}",
        "",
        "Confusion Matrix (Test):",
        f"- TN: {test_metrics['confusion_matrix']['tn']}",
        f"- FP: {test_metrics['confusion_matrix']['fp']}",
        f"- FN: {test_metrics['confusion_matrix']['fn']}",
        f"- TP: {test_metrics['confusion_matrix']['tp']}",
        "",
    ]
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"Selected model: {selected_model_type}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Saved artifacts to: {output_dir.resolve()}")
    print(f"Report: {report_path.resolve()}")

if __name__ == "__main__":
    main()
   
