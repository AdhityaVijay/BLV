"""Config loading. Every module reads tunables from config.yaml via here, so
there is exactly one place to change a threshold.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

_cache: dict[str, Any] | None = None


def load_config(path: str | Path | None = None, reload: bool = False) -> dict[str, Any]:
    global _cache
    if _cache is not None and not reload and path is None:
        return _cache
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_project_root"] = str(PROJECT_ROOT)
    if path is None:
        _cache = cfg
    return cfg


def resolve_path(relative: str) -> Path:
    """Resolve a config-relative path against the project root, so scripts work
    regardless of the directory they are launched from."""
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p
