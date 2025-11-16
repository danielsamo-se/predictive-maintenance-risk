import pandas as pd
import pytest

from pmrisk.quality.checks import (
    assert_expected_columns,
    assert_monotonic_cycle_per_engine,
    assert_no_duplicates,
    assert_sorted_by_engine_cycle,
)


def test_no_duplicates_pass():
    df = pd.DataFrame({"engine_id": [1, 1, 2], "cycle": [1, 2, 1]})
    assert_no_duplicates(df, keys=["engine_id", "cycle"])


def test_duplicates_fail():
    df = pd.DataFrame({"engine_id": [1, 1], "cycle": [1, 1]})
    with pytest.raises(ValueError, match="duplicate"):
        assert_no_duplicates(df, keys=["engine_id", "cycle"])


def test_monotonic_cycle_fail():
    df = pd.DataFrame({"engine_id": [1, 1, 1], "cycle": [1, 3, 2]})
    with pytest.raises(ValueError, match="monotonic"):
        assert_monotonic_cycle_per_engine(df)


def test_good_case():
    df = pd.DataFrame(
        {
            "engine_id": [1, 1, 2, 2],
            "cycle": [1, 2, 1, 2],
            "sensor_1": [1.0, 2.0, 3.0, 4.0],
        }
    )
    assert_expected_columns(df, ["engine_id", "cycle", "sensor_1"])
    assert_no_duplicates(df, keys=["engine_id", "cycle"])
    assert_monotonic_cycle_per_engine(df)
    assert_sorted_by_engine_cycle(df)
