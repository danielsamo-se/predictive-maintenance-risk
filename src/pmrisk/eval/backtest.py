"""
Backtest for C-MAPSS FD001 using engine-separated expanding folds to prevent leakage
Per fold, tune the model and decision threshold on a validation split, then report test metrics
"""

import math
from pathlib import Path

import lightgbm as lgb
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pmrisk.eval.calibration import compute_calibration
from pmrisk.eval.errors import summarize_errors
from pmrisk.eval.metrics import compute_metrics
from pmrisk.eval.thresholds import select_threshold
from pmrisk.features.contracts import validate_tabular_features
from pmrisk.features.tabular import build_tabular_features


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_expanding_engine_folds(
    engine_ids: list[int],
    n_folds: int,
    train_min_engines: int,
    test_engines_per_fold: int,
) -> list[dict]:

    sorted_ids = sorted(engine_ids)
    folds = []
    
    for i in range(n_folds):
        train_end = train_min_engines + i * test_engines_per_fold
        test_start = train_end
        test_end = test_start + test_engines_per_fold
        
        if test_end > len(sorted_ids):
            break
        
        train_engines = sorted_ids[:train_end]
        test_engines = sorted_ids[test_start:test_end]
        
        folds.append({
            "fold": i,
            "train_engines": train_engines,
            "test_engines": test_engines,
        })
    
    return folds


def split_train_val_engines(
    train_engines: list[int],
    val_ratio: float,
) -> tuple[list[int], list[int]]:
    n = len(train_engines)
    if n <= 1:
        return train_engines, []
    
    n_val = max(1, math.ceil(val_ratio * n))
    
    train = train_engines[:-n_val]
    val = train_engines[-n_val:]
    
    return train, val


def run_backtest() -> str:
    base_cfg = load_config("configs/base.yaml")
    feat_cfg = load_config("configs/features.yaml")
    thresh_cfg = load_config("configs/thresholds.yaml")
    backtest_cfg = load_config("configs/backtest.yaml")
    
    window_l = base_cfg["window_l"]
    random_seed = base_cfg["random_seed"]
    horizon_n = base_cfg["horizon_n"]
    
    processed_dir = Path(base_cfg["data_processed_dir"])
    df_full = pd.read_parquet(processed_dir / "fd001_train.parquet")
    df_labels_full = pd.read_parquet(processed_dir / "fd001_train_labeled.parquet")
    
    label_cols = ["engine_id", "cycle", "label"]
    if "remaining" in df_labels_full.columns:
        label_cols.append("remaining")
    df_labels = df_labels_full[label_cols]
    
    feat_spec = {
        "windows": feat_cfg["windows"],
        "stats": feat_cfg["stats"],
        "lags": feat_cfg["lags"],
        "deltas": feat_cfg["deltas"],
        "include_slopes": feat_cfg["include_slopes"],
    }
    
    df_feat = build_tabular_features(df_full, feat_spec, window_l, filter_after=True)
    validate_tabular_features(df_feat, window_l)
    
    df_feat = df_feat.merge(df_labels, on=["engine_id", "cycle"], how="inner")
    
    all_engine_ids = sorted(df_feat["engine_id"].unique().tolist())
    
    folds = make_expanding_engine_folds(
        all_engine_ids,
        backtest_cfg["n_folds"],
        backtest_cfg["train_min_engines"],
        backtest_cfg["test_engines_per_fold"],
    )
    
    exclude_cols = {"engine_id", "cycle", "label", "remaining", "failure_cycle"}
    feature_columns = sorted([c for c in df_feat.columns if c not in exclude_cols])

    
    model_candidates = backtest_cfg["model_candidates"]
    
    results = []
    
    for fold_info in folds:
        fold_idx = fold_info["fold"]
        train_engines_all = fold_info["train_engines"]
        test_engines = fold_info["test_engines"]
        
        train_engines, val_engines = split_train_val_engines(
            train_engines_all, backtest_cfg["val_ratio_within_train"]
        )

        if len(val_engines) == 0:
            raise ValueError("val_engines is empty. Increase train_min_engines or val_ratio_within_train")
        
        X_train = df_feat[df_feat["engine_id"].isin(train_engines)][feature_columns]
        y_train = df_feat[df_feat["engine_id"].isin(train_engines)]["label"]
        X_val = df_feat[df_feat["engine_id"].isin(val_engines)][feature_columns]
        y_val = df_feat[df_feat["engine_id"].isin(val_engines)]["label"]
        X_test = df_feat[df_feat["engine_id"].isin(test_engines)][feature_columns]
        y_test = df_feat[df_feat["engine_id"].isin(test_engines)]["label"]
        
        best_model_type = None
        best_val_pr_auc = -1
        best_proba_val = None
        best_proba_test = None
        
        for candidate in model_candidates:
            if candidate == "logreg":
                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=random_seed
                    ))
                ])
            elif candidate == "lightgbm":
                model = lgb.LGBMClassifier(
                    random_state=random_seed,
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    class_weight="balanced",
                    verbosity=-1,
                )
            else:
                continue
            
            model.fit(X_train, y_train)
            
            proba_val = model.predict_proba(X_val)[:, 1]
            val_metrics = compute_metrics(y_val, proba_val, threshold=0.5)
            val_pr_auc = val_metrics["pr_auc"]
            
            if val_pr_auc > best_val_pr_auc:
                best_val_pr_auc = val_pr_auc
                best_model_type = candidate
                best_proba_val = proba_val
                best_proba_test = model.predict_proba(X_test)[:, 1]

        if best_proba_val is None or best_proba_test is None or best_model_type is None:     
            raise ValueError(f"No valid model candidate selected in fold {fold_idx}")  
        
        threshold = select_threshold(
            y_val,
            best_proba_val,
            thresh_cfg["target_recall"],
            thresh_cfg["min_precision"],
        )
        
        test_metrics = compute_metrics(y_test, best_proba_test, threshold)
        
        test_df_base = df_feat[df_feat["engine_id"].isin(test_engines)][["engine_id", "cycle", "label"]].copy()
        if "remaining" in df_labels.columns:
            test_df_remaining = df_labels[df_labels["engine_id"].isin(test_engines)][["engine_id", "cycle", "remaining"]]
            test_df_base = test_df_base.merge(test_df_remaining, on=["engine_id", "cycle"], how="left")
        
        test_df_base["y_proba"] = best_proba_test
        test_df_base["y_pred"] = (best_proba_test >= threshold).astype(int)
        
        errors = summarize_errors(
            test_df_base,
            horizon_n=horizon_n,
            early_alarm_slack=backtest_cfg["early_alarm_slack"],
            top_k=20,
        )
        
        cal = compute_calibration(
            y_test,
            best_proba_test,
            n_bins=backtest_cfg["calibration_bins"],
            strategy=backtest_cfg["calibration_strategy"],
        )
        
        results.append({
            "fold": fold_idx,
            "train_count": len(train_engines),
            "val_count": len(val_engines),
            "test_count": len(test_engines),
            "model_type": best_model_type,
            "threshold": threshold,
            "val_pr_auc": best_val_pr_auc,
            "test_pr_auc": test_metrics["pr_auc"],
            "test_roc_auc": test_metrics["roc_auc"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
            "test_cm": test_metrics["confusion_matrix"],
            "brier_score": cal["brier_score"],
            "errors": errors,
        })
    
    report_lines = []
    report_lines.append("Expanding Engine Backtest Results")
    report_lines.append("")

    report_lines.append("Configuration")
    report_lines.append(f"- n_folds: {backtest_cfg['n_folds']}")
    report_lines.append(f"- test_engines_per_fold: {backtest_cfg['test_engines_per_fold']}")
    report_lines.append(f"- train_min_engines: {backtest_cfg['train_min_engines']}")
    report_lines.append(f"- val_ratio_within_train: {backtest_cfg['val_ratio_within_train']}")
    report_lines.append(f"- model_candidates: {model_candidates}")
    report_lines.append(f"- target_recall: {thresh_cfg['target_recall']}")
    report_lines.append(f"- min_precision: {thresh_cfg['min_precision']}")
    report_lines.append("")

    if not results:
        report_lines.append("No folds were generated (check backtest config vs. available engines).")
        return "\n".join(report_lines)

    report_lines.append("Fold results")
    report_lines.append("")

    for res in results:
        cm = res["test_cm"]
        errors = res["errors"]

        report_lines.append(f"Fold {res['fold']}:")
        report_lines.append(
            f"- engines: train={res['train_count']}, val={res['val_count']}, test={res['test_count']}"
        )
        report_lines.append(f"- selected_model: {res['model_type']}")
        report_lines.append(f"- threshold: {res['threshold']:.4f}")
        report_lines.append(f"- val_pr_auc: {res['val_pr_auc']:.4f}")

        report_lines.append("  Test metrics:")
        report_lines.append(f"  - pr_auc: {res['test_pr_auc']:.4f}")
        report_lines.append(f"  - roc_auc: {res['test_roc_auc']:.4f}")
        report_lines.append(f"  - precision: {res['test_precision']:.4f}")
        report_lines.append(f"  - recall: {res['test_recall']:.4f}")
        report_lines.append(f"  - f1: {res['test_f1']:.4f}")
        report_lines.append(f"  - brier_score: {res['brier_score']:.4f}")
        report_lines.append(
            f"  - confusion_matrix: tn={cm['tn']}, fp={cm['fp']}, fn={cm['fn']}, tp={cm['tp']}"
        )
        
        report_lines.append("  Top false positives (max 5):")
        for fp in errors["false_positives"][:5]:
            line = f"    - engine_id={fp['engine_id']}, cycle={fp['cycle']}, y_proba={fp['y_proba']:.3f}"
            if "remaining" in fp:
                line += f", remaining={fp['remaining']}"
            if "fp_kind" in fp:
                line += f", fp_kind={fp['fp_kind']}"
            report_lines.append(line)
        
        report_lines.append("  Top false negatives (max 5):")
        for fn in errors["false_negatives"][:5]:
            line = f"    - engine_id={fn['engine_id']}, cycle={fn['cycle']}, y_proba={fn['y_proba']:.3f}"
            if "remaining" in fn:
                line += f", remaining={fn['remaining']}"
            report_lines.append(line)
        
        report_lines.append("")

    return "\n".join(report_lines)


if __name__ == "__main__":
    report = run_backtest()
    
    output_dir = Path("reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "backtest_calibration.md"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(output_path.resolve())
