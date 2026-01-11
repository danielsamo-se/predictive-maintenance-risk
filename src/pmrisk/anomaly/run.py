"""Run anomaly detection on FD001 data and save scores + a report"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pmrisk.anomaly.isoforest import fit_isoforest, score_isoforest
from pmrisk.features.tabular import build_tabular_features
from pmrisk.split.splitter import split_engine_ids


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _validate_contamination(contamination: float) -> None:
    if not (0.0 < float(contamination) <= 0.5):
        raise ValueError(f"Invalid contamination={contamination}. Expected 0 < contamination <= 0.5")


def _make_scored_df(df_split: pd.DataFrame, scores: np.ndarray, split_name: str) -> pd.DataFrame:
    out = df_split[["engine_id", "cycle", "label", "remaining"]].copy()
    out["anomaly_score"] = scores.astype(np.float64)
    out["split"] = split_name
    return out[["engine_id", "cycle", "label", "remaining", "anomaly_score", "split"]]


def _fmt(x: float | None, digits: int = 4) -> str:
    if x is None:
        return "N/A"
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return "N/A"
    return f"{float(x):.{digits}f}"


def _topk_stats(df: pd.DataFrame, k: float) -> dict:
    n = len(df)
    if n == 0:
        return {"m": 0, "precision": None, "lift": None, "capture": None}

    m = max(1, int(math.ceil(k * n)))
    d = df.sort_values("anomaly_score", ascending=False)
    top = d.head(m)

    pos = int(df["label"].sum())
    base_rate = float(df["label"].mean())

    top_pos = int(top["label"].sum())
    precision = float(top_pos / m)
    lift = float(precision / base_rate) if base_rate > 0 else None
    capture = float(top_pos / pos) if pos > 0 else None

    return {"m": m, "precision": precision, "lift": lift, "capture": capture}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run anomaly detection")
    parser.add_argument("--config", default="configs/anomaly.yaml", help="Path to anomaly config")
    args = parser.parse_args()

    base_cfg = _load_yaml("configs/base.yaml")
    split_cfg = _load_yaml("configs/split.yaml")
    anomaly_cfg = _load_yaml(args.config)
    features_cfg = _load_yaml("configs/features.yaml")

    _validate_contamination(anomaly_cfg["contamination"])

    processed_dir = Path(base_cfg.get("data_processed_dir", "data/processed"))
    data_path = processed_dir / "fd001_train_labeled.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    df_labeled = pd.read_parquet(data_path)

    print("Building tabular features")

    feat_spec = {
        "windows": features_cfg["windows"],
        "stats": features_cfg["stats"],
        "lags": features_cfg["lags"],
        "deltas": features_cfg["deltas"],
        "include_slopes": features_cfg.get("include_slopes", False),
    }

    df_feat = build_tabular_features(
        df_labeled,
        spec=feat_spec,
        window_l=base_cfg["window_l"],
        filter_after=True,
    )

    df_full = df_feat.merge(
        df_labeled[["engine_id", "cycle", "label", "remaining", "failure_cycle"]],
        on=["engine_id", "cycle"],
        how="inner",
    )

    exclude_cols = {"engine_id", "cycle", "label", "failure_cycle", "remaining"}
    feature_columns = sorted([c for c in df_full.columns if c not in exclude_cols])
    if not feature_columns:
        raise ValueError("No feature columns found after excluding key/label columns.")

    print(f"Feature columns: {len(feature_columns)}")

    all_engine_ids = sorted(df_full["engine_id"].unique())
    splits = split_engine_ids(
        all_engine_ids,
        train_ratio=split_cfg["train_ratio"],
        val_ratio=split_cfg["val_ratio"],
        test_ratio=split_cfg["test_ratio"],
        seed=split_cfg["seed"],
    )

    train_engines = set(splits["train"])
    val_engines = set(splits["val"])
    test_engines = set(splits["test"])

    df_train = df_full[df_full["engine_id"].isin(train_engines)].copy()
    df_val = df_full[df_full["engine_id"].isin(val_engines)].copy()
    df_test = df_full[df_full["engine_id"].isin(test_engines)].copy()

    print(f"Train: {len(df_train)} samples")
    print(f"Val: {len(df_val)} samples")
    print(f"Test: {len(df_test)} samples")

    X_train = df_train[feature_columns].to_numpy(dtype=np.float64)
    X_val = df_val[feature_columns].to_numpy(dtype=np.float64)
    X_test = df_test[feature_columns].to_numpy(dtype=np.float64)

    for name, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
        n_nan = int(np.isnan(X).sum())
        n_inf = int(np.isinf(X).sum())
        if n_nan or n_inf:
            print(f"Non-finite before impute ({name}): nan={n_nan}, inf={n_inf}")

    X_train = np.where(np.isinf(X_train), np.nan, X_train)
    X_val = np.where(np.isinf(X_val), np.nan, X_val)
    X_test = np.where(np.isinf(X_test), np.nan, X_test)

    col_median = np.nanmedian(X_train, axis=0)
    if np.isnan(col_median).any():
        bad_cols = [feature_columns[i] for i in np.where(np.isnan(col_median))[0].tolist()]
        raise ValueError(f"Train median is NaN for columns: {bad_cols[:20]}")

    X_train = np.where(np.isnan(X_train), col_median, X_train)
    X_val = np.where(np.isnan(X_val), col_median, X_val)
    X_test = np.where(np.isnan(X_test), col_median, X_test)

    for name, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
        if not np.isfinite(X).all():
            raise ValueError(f"Non-finite values found in {name} features after imputation (NaN/Inf)")

    print("Fitting IsolationForest")
    model = fit_isoforest(
        X_train,
        contamination=float(anomaly_cfg["contamination"]),
        seed=int(anomaly_cfg["random_seed"]),
    )

    print("Scoring")
    train_scores = score_isoforest(model, X_train)
    val_scores = score_isoforest(model, X_val)
    test_scores = score_isoforest(model, X_test)

    df_scores = pd.concat(
        [
            _make_scored_df(df_train, train_scores, "train"),
            _make_scored_df(df_val, val_scores, "val"),
            _make_scored_df(df_test, test_scores, "test"),
        ],
        ignore_index=True,
    )

    df_scores["engine_id"] = df_scores["engine_id"].astype(int)
    df_scores["cycle"] = df_scores["cycle"].astype(int)
    df_scores["label"] = df_scores["label"].astype(int)

    output_path = Path(anomaly_cfg["output_scores_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_scores.to_parquet(output_path, index=False)

    print(f"Saved anomaly scores to {output_path}")
    print(f"Total scored samples: {len(df_scores)}")

    report_path = Path("reports/anomaly_eval.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    contamination = anomaly_cfg["contamination"]
    seed = anomaly_cfg["random_seed"]
    horizon_n = base_cfg.get("horizon_n", "N/A")
    window_l = base_cfg.get("window_l", "N/A")
    feature_set_version = base_cfg.get("feature_set_version", "N/A")

    lines: list[str] = []
    lines.append("Anomaly Baseline (IsolationForest)")
    lines.append("")
    lines.append("Config:")
    lines.append(f"- contamination: {contamination}")
    lines.append(f"- seed: {seed}")
    lines.append(f"- horizon_n: {horizon_n}")
    lines.append(f"- window_l: {window_l}")
    lines.append(f"- feature_set_version: {feature_set_version}")
    lines.append("- fit: train only")
    lines.append("- scoring: val/test")
    lines.append("")
    lines.append("Notes:")
    lines.append("- top_1pct_precision = positive rate within the top 1% anomaly scores")
    lines.append("- top_1pct_lift = top_1pct_precision / base_rate")
    lines.append("- top_1pct_capture = fraction of all positives captured in the top 1%")
    lines.append("")
    lines.append("Metrics")
    lines.append("")

    for split_name in ["val", "test"]:
        d = df_scores[df_scores["split"] == split_name].copy()
        n = len(d)
        pos = int(d["label"].sum())
        base_rate = float(d["label"].mean()) if n > 0 else float("nan")

        top1 = _topk_stats(d, 0.01)
        top5 = _topk_stats(d, 0.05)

        lines.append(f"{split_name.capitalize()}:")
        lines.append(f"- n: {n}")
        lines.append(f"- positives: {pos}")
        lines.append(f"- base_rate: {_fmt(base_rate, 4)}")
        lines.append(f"- top_1pct_m: {top1['m']}")
        lines.append(f"- top_1pct_precision: {_fmt(top1['precision'], 4)}")
        lines.append(f"- top_1pct_lift: {_fmt(top1['lift'], 4)}")
        lines.append(f"- top_1pct_capture: {_fmt(top1['capture'], 4)}")
        lines.append(f"- top_5pct_m: {top5['m']}")
        lines.append(f"- top_5pct_precision: {_fmt(top5['precision'], 4)}")
        lines.append(f"- top_5pct_lift: {_fmt(top5['lift'], 4)}")
        lines.append(f"- top_5pct_capture: {_fmt(top5['capture'], 4)}")
        lines.append("")

        lines.append(f"Top 10 anomalies ({split_name.upper()}):")
        top10 = d.sort_values("anomaly_score", ascending=False).head(10)
        for i, row in enumerate(top10.itertuples(index=False), start=1):
            lines.append(
                f"- {i}. engine_id={int(row.engine_id)}, cycle={int(row.cycle)}, "
                f"label={int(row.label)}, remaining={int(row.remaining)}, "
                f"anomaly_score={float(row.anomaly_score):.6f}"
            )
        lines.append("")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
