import pandas as pd


def build_tabular_features(
    df: pd.DataFrame,
    spec: dict,
    window_l: int,
    filter_after: bool = True,
    id_col: str = "engine_id",
    time_col: str = "cycle",
) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values([id_col, time_col]).reset_index(drop=True)

    prefixes = ("op_setting_", "sensor_")
    signal_cols = [
        col
        for col in df.columns
        if col.startswith(prefixes) and pd.api.types.is_numeric_dtype(df[col])
    ]

    if not signal_cols:
        raise ValueError(
            "No signal columns found. Expected columns starting with "
            "'op_setting_' or 'sensor_'."
        )

    feature_dfs = []

    for sig_col in signal_cols:
        for window in spec.get("windows", []):
            for stat in spec.get("stats", []):
                feat_name = f"{sig_col}__roll{window}__{stat}"
                feature_dfs.append(
                    df.groupby(id_col)[sig_col]
                    .rolling(window=window, min_periods=window)
                    .agg(stat)
                    .reset_index(level=0, drop=True)
                    .rename(feat_name)
                )

        for lag in spec.get("lags", []):
            feat_name = f"{sig_col}__lag{lag}"
            feature_dfs.append(
                df.groupby(id_col)[sig_col].shift(lag).rename(feat_name)
            )

        for delta in spec.get("deltas", []):
            feat_name = f"{sig_col}__delta{delta}"
            feature_dfs.append(
                df.groupby(id_col)[sig_col].diff(delta).rename(feat_name)
            )

    result = pd.concat([df[[id_col, time_col]]] + feature_dfs, axis=1)

    min_cycle = max(
    window_l,
    max(spec.get("windows", [1])),
    max(spec.get("lags", [0])) + 1,
    max(spec.get("deltas", [0])) + 1,
    ) 

    if filter_after:
        result = result[result[time_col] >= min_cycle].copy()

    result = result.sort_values([id_col, time_col]).reset_index(drop=True)

    feature_cols = sorted([c for c in result.columns if c not in [id_col, time_col]])
    result = result[[id_col, time_col] + feature_cols]

    return result

