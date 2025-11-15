from pathlib import Path

import pandas as pd

from pmrisk.config import settings


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
        sep=" ",
        header=None,
        names=columns,
        dtype=dtype_dict,
        engine="pyarrow",
    )
    
    df = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    
    df.to_parquet(output_path, index=False)


if __name__ == "__main__":
    main()
