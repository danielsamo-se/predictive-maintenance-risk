import pytest
import torch

from pmrisk.models.sequence_gru import SequenceGRU


def test_forward_shape_ok() -> None:
    model = SequenceGRU(n_features=14, window_l=50)
    x = torch.randn(4, 50, 14)

    logits = model(x)

    assert logits.shape == (4, 1)


def test_forward_raises_on_wrong_shape() -> None:
    model = SequenceGRU(n_features=14, window_l=50)

    with pytest.raises(ValueError, match=r"Expected x\.ndim == 3"):
        model(torch.randn(4, 50))

    with pytest.raises(ValueError, match=r"Expected L == 50"):
        model(torch.randn(4, 40, 14))

    with pytest.raises(ValueError, match=r"Expected F == 14"):
        model(torch.randn(4, 50, 10))


