
import pytest

torch = pytest.importorskip("torch")

from pmrisk.models.sequence_cnn import SequenceCNN


def test_forward_shape_ok() -> None:
    model = SequenceCNN(n_features=14, window_l=50)
    x = torch.randn(4, 50, 14)

    logits = model(x)

    assert logits.shape == (4, 1)


def test_forward_raises_on_wrong_shape() -> None:
    model = SequenceCNN(n_features=14, window_l=50)

    x_wrong_ndim = torch.randn(4, 50)
    with pytest.raises(ValueError, match=r"Expected x\.ndim == 3"):
        model(x_wrong_ndim)

    x_wrong_l = torch.randn(4, 40, 14)
    with pytest.raises(ValueError, match=r"Expected L == 50"):
        model(x_wrong_l)

    x_wrong_f = torch.randn(4, 50, 10)
    with pytest.raises(ValueError, match=r"Expected F == 14"):
        model(x_wrong_f)

