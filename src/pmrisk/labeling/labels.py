"""Label FD001 cycles with risk classes for the prediction horizon; optionally save the labeled dataset as parquet"""
from pathlib import Path

import pandas as pd

from pmrisk.config import settings


def build_labeled_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add failure_cycle/remaining/label (label=1 iff 0 < remaining <= horizon_n) and drop cycles < window_l"""
    if "engine_id" not in df.columns or "cycle" not in df.columns:
        raise ValueError("Missing required columns: engine_id or cycle")
    
    df = df.copy()
    
    failure_cycle = df.groupby("engine_id")["cycle"].transform("max")
    df["failure_cycle"] = failure_cycle
    df["remaining"] = df["failure_cycle"] - df["cycle"]
    df["label"] = ((df["remaining"] > 0) & (df["remaining"] <= settings.horizon_n)).astype(int)
    
    df = df[df["cycle"] >= settings.window_l].copy()
    df = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    
    return df


def main():
    input_path = Path(settings.data_processed_dir) / "fd001_train.parquet"
    output_path = Path(settings.data_processed_dir) / "fd001_train_labeled.parquet"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_parquet(input_path)
    df_labeled = build_labeled_df(df)
    df_labeled.to_parquet(output_path, index=False)


if __name__ == "__main__":
    main()
