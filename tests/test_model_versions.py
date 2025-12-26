from pathlib import Path

import pytest

from pmrisk.models.model_versions import (
    get_active_version,
    make_version_dir,
    read_metadata,
    set_active_version,
    write_metadata,
)


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
