import torch

from pmrisk.models.train_sequence import select_threshold_for_policy


def test_select_threshold_for_policy_picks_threshold_with_min_precision() -> None:
    y = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.float32)
    s = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.4, 0.2], dtype=torch.float32)

    thr = select_threshold_for_policy(y, s, target_recall=2 / 3, min_precision=1.0)
    assert abs(thr - 0.8) < 1e-6


def test_select_threshold_for_policy_raises_if_impossible() -> None:
    y = torch.tensor([1, 1, 0], dtype=torch.float32)
    s = torch.tensor([0.9, 0.8, 0.85], dtype=torch.float32)

    try:
        select_threshold_for_policy(y, s, target_recall=1.0, min_precision=1.0)
        assert False, "expected ValueError"
    except ValueError:
        pass
