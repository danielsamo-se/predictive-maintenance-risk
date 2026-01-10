"""Project config loaded from configs"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


def _default_config_path() -> Path:
   
    env_path = os.getenv("PMRISK_CONFIG_PATH")
    if env_path:
        return Path(env_path)

    env_root = os.getenv("PMRISK_PROJECT_ROOT")
    if env_root:
        return Path(env_root) / "configs" / "base.yaml"

    return Path("configs") / "base.yaml"


_CONFIG_PATH = _default_config_path()


def _load_yaml_config(path: Path = _CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found: {path} "
        )

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be YAML ")
    return data


class Settings(BaseModel):
    horizon_n: int
    window_l: int
    feature_set_version: str
    random_seed: int
    data_raw_dir: str
    data_processed_dir: str
    models_production_dir: str


settings = Settings(**_load_yaml_config())
