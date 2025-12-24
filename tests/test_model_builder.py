import pytest

from pmrisk.models.model_factory import build_sequence_model
from pmrisk.models.sequence_cnn import SequenceCNN


def test_build_sequence_model_cnn_ok() -> None:
    hparams = {"model_type": "cnn", "n_features": 14, "window_l": 50}

    model = build_sequence_model(hparams)

    assert isinstance(model, SequenceCNN)
    assert model.n_features == 14
    assert model.window_l == 50


def test_build_sequence_model_cnn_with_optional_params() -> None:
    hparams = {
        "model_type": "cnn",
        "n_features": 14,
        "window_l": 50,
        "hidden_channels": 64,
        "kernel_size": 5,
        "dropout_p": 0.1,
    }

    model = build_sequence_model(hparams)

    assert isinstance(model, SequenceCNN)
    assert model.n_features == 14
    assert model.window_l == 50


def test_build_sequence_model_unknown_type_raises() -> None:
    hparams = {"model_type": "weird", "n_features": 14, "window_l": 50}

    with pytest.raises(ValueError, match="Unknown model_type"):
        build_sequence_model(hparams)


def test_build_sequence_model_missing_required_key_raises() -> None:
    with pytest.raises(KeyError, match="Missing required key: n_features"):
        build_sequence_model({"model_type": "cnn", "window_l": 50})

    with pytest.raises(KeyError, match="Missing required key: window_l"):
        build_sequence_model({"model_type": "cnn", "n_features": 14})

    with pytest.raises(KeyError, match="Missing required key: model_type"):
        build_sequence_model({"n_features": 14, "window_l": 50})


