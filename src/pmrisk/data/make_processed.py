"""Parse raw CMAPSS FD001 text files into a typed and sorted parquet dataset"""
from pathlib import Path

import pandas as pd

from pmrisk.config import settings
from pmrisk.quality.checks import (
    assert_expected_columns,
    assert_monotonic_cycle_per_engine,
    assert_no_duplicates,
    assert_sorted_by_engine_cycle,
)


def main():
    input_path = Path(settings.data_raw_dir) / "FD001" / "train_FD001.txt"
    output_path = Path(settings.data_processed_dir) / "fd001_train.parquet"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    columns = (
        ["engine_id", "cycle"]
        + [f"op_setting_{i}" for i in range(1, 4)]
        + [f"sensor_{i}" for i in range(1, 22)]
    )
    
    dtype_dict = {
        "engine_id": "int64",
        "cycle": "int64",
    }
    for col in columns[2:]:
        dtype_dict[col] = "float64"
    
    df = pd.read_csv(
    input_path,
    sep=r"\s+",
    header=None,
    names=columns,
    dtype=dtype_dict,
    )
    
    df = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    
    assert_expected_columns(df, columns)
    assert_no_duplicates(df, keys=["engine_id", "cycle"])
    assert_monotonic_cycle_per_engine(df)
    assert_sorted_by_engine_cycle(df)
    
    df.to_parquet(output_path, index=False)


if __name__ == "__main__":
    main()
