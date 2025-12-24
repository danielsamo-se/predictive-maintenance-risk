"""Build sequence models from a hyperparameter dictionary"""

from __future__ import annotations

from pmrisk.models.sequence_cnn import SequenceCNN


def build_sequence_model(hparams: dict):
    required_keys = ("model_type", "n_features", "window_l")
    for key in required_keys:
        if key not in hparams:
            raise KeyError(f"Missing required key: {key}")

    model_type = hparams["model_type"]
    n_features = hparams["n_features"]
    window_l = hparams["window_l"]

    if model_type == "cnn":
        cnn_kwargs = {"n_features": n_features, "window_l": window_l}

        if "hidden_channels" in hparams:
            cnn_kwargs["hidden_channels"] = hparams["hidden_channels"]
        if "kernel_size" in hparams:
            cnn_kwargs["kernel_size"] = hparams["kernel_size"]
        if "dropout_p" in hparams:
            cnn_kwargs["dropout_p"] = hparams["dropout_p"]

        return SequenceCNN(**cnn_kwargs)

    raise ValueError(f"Unknown model_type: {model_type}")
