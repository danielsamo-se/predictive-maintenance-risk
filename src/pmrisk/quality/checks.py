"""Basic data quality checks: column presence, uniqueness, monotonicity, sorting"""
import pandas as pd


def assert_expected_columns(df: pd.DataFrame, expected_cols: list[str]) -> None:
    missing = set(expected_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    extra = set(df.columns) - set(expected_cols)
    if extra:
        raise ValueError(f"Unexpected columns: {extra}")


def assert_no_duplicates(df: pd.DataFrame, keys: list[str]) -> None:
    duplicates = df[df.duplicated(subset=keys, keep=False)]
    if not duplicates.empty:
        raise ValueError(f"Found {len(duplicates)} duplicate rows on keys {keys}")


def assert_monotonic_cycle_per_engine(df: pd.DataFrame) -> None:
    for engine_id, group in df.groupby("engine_id"):
        cycles = group["cycle"].values
        if not all(cycles[i] < cycles[i + 1] for i in range(len(cycles) - 1)):
            raise ValueError(f"Cycle not strictly monotonic for engine_id={engine_id}")


def assert_sorted_by_engine_cycle(df: pd.DataFrame) -> None:
    if not df.equals(df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)):
        raise ValueError("DataFrame not sorted by (engine_id, cycle)")
