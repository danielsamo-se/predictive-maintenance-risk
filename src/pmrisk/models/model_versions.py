"""Helper functions to save models with versions and a metadata file"""

from __future__ import annotations

import json
from pathlib import Path


def _get_root(root: Path | None) -> Path:
    return Path("models") / "production" if root is None else root


def make_version_dir(model_name: str, version: str, root: Path | None = None) -> Path:
    base = _get_root(root)
    version_dir = base / model_name / version
    version_dir.mkdir(parents=True, exist_ok=True)
    return version_dir


def write_metadata(dir_path: Path, metadata: dict) -> None:
    metadata_path = dir_path / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def read_metadata(dir_path: Path) -> dict:
    metadata_path = dir_path / "metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_active_version(model_name: str, version: str, root: Path | None = None) -> None:
    base = _get_root(root)
    model_dir = base / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    active_path = model_dir / "ACTIVE"
    with open(active_path, "w", encoding="utf-8") as f:
        f.write(f"{version}\n")


def get_active_version(model_name: str, root: Path | None = None) -> str:
    base = _get_root(root)
    active_path = base / model_name / "ACTIVE"
    with open(active_path, "r", encoding="utf-8") as f:
        return f.read().strip()
