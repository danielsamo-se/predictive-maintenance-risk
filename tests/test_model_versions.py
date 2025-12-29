from pathlib import Path

import pytest
import torch

from pmrisk.models.model_builder import build_sequence_model
from pmrisk.models.model_versions import (
    get_active_version,
    load_active_sequence_model,
    make_version_dir,
    read_metadata,
    save_sequence_model_production,
    set_active_version,
    write_metadata,
)
from pmrisk.models.sequence_cnn import SequenceCNN


def test_make_version_dir_creates_path(tmp_path: Path) -> None:
    version_dir = make_version_dir("my_model", "v1", root=tmp_path)

    assert version_dir.exists()
    assert version_dir.is_dir()
    assert version_dir == tmp_path / "my_model" / "v1"


def test_write_and_read_metadata_roundtrip(tmp_path: Path) -> None:
    metadata = {"model_type": "cnn", "n_features": 14, "window_l": 50, "threshold": 0.42}

    version_dir = tmp_path / "test_model" / "v1"
    version_dir.mkdir(parents=True)

    write_metadata(version_dir, metadata)
    loaded = read_metadata(version_dir)

    assert loaded == metadata


def test_set_and_get_active_version_roundtrip(tmp_path: Path) -> None:
    set_active_version("my_model", "v2", root=tmp_path)

    assert (tmp_path / "my_model" / "ACTIVE").exists()
    assert get_active_version("my_model", root=tmp_path) == "v2"


def test_get_active_version_raises_if_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        get_active_version("nonexistent_model", root=tmp_path)


def test_save_and_load_active_sequence_model_roundtrip(tmp_path: Path) -> None:
    hparams = {"model_type": "cnn", "n_features": 3, "window_l": 10, "hidden_channels": 8}

    model = build_sequence_model(hparams)
    state_dict = model.state_dict()

    metadata = {"hparams": hparams, "model_type": "cnn", "some_other_field": "test_value"}

    version_dir = save_sequence_model_production(
        model_name="seq",
        version="v1",
        state_dict=state_dict,
        metadata=metadata,
        root=tmp_path,
        set_active=True,
    )

    assert version_dir.exists()
    assert (version_dir / "metadata.json").exists()
    assert (version_dir / "model.pt").exists()
    assert (tmp_path / "seq" / "ACTIVE").exists()

    loaded_model, loaded_metadata = load_active_sequence_model("seq", root=tmp_path)

    assert loaded_metadata["hparams"] == hparams
    assert loaded_metadata["model_type"] == "cnn"
    assert loaded_metadata["some_other_field"] == "test_value"
    assert isinstance(loaded_model, SequenceCNN)

    x = torch.randn(2, 10, 3)
    with torch.no_grad():
        output = loaded_model(x)

    assert output.shape == (2, 1)
