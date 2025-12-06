"""Project config loaded from configs"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

_PROJECT_ROOT = Path(__file__).resolve().parents[2]  
_CONFIG_PATH = _PROJECT_ROOT / "configs" / "base.yaml"


def _load_yaml_config(path: Path = _CONFIG_PATH) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a YAML mapping.")
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
