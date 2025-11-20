import pandas as pd

from pmrisk.config import settings
from pmrisk.labeling.labels import build_labeled_df


def test_labeling_boundaries():
    L = settings.window_l
    N = settings.horizon_n
    
    df = pd.DataFrame(
        {
            "engine_id": [1, 1, 1, 1, 2, 2, 2],
            "cycle": [L - 1, L, L + 1, L + N + 1, L, L + 1, L + N],
        }
    )
    
    df_labeled = build_labeled_df(df)
    
    engine1 = df_labeled[df_labeled["engine_id"] == 1].sort_values("cycle")
    engine2 = df_labeled[df_labeled["engine_id"] == 2].sort_values("cycle")
    
    assert engine1["failure_cycle"].iloc[0] == L + N + 1
    assert engine2["failure_cycle"].iloc[0] == L + N
    
    row1_remaining_0 = engine1[engine1["remaining"] == 0]
    assert len(row1_remaining_0) > 0
    assert row1_remaining_0["label"].iloc[0] == 0
    
    row1_remaining_n = engine1[engine1["remaining"] == N]
    assert len(row1_remaining_n) > 0
    assert row1_remaining_n["label"].iloc[0] == 1
    
    row1_remaining_n_plus_1 = engine1[engine1["remaining"] == N + 1]
    assert len(row1_remaining_n_plus_1) > 0
    assert row1_remaining_n_plus_1["label"].iloc[0] == 0
    
    assert df_labeled["cycle"].min() >= L
    assert len(df_labeled) == 6
