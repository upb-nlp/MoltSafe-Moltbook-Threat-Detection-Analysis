from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


@lru_cache(maxsize=1)
def root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError("could not locate repository root containing pyproject.toml")


def config(*parts: str) -> Path:
    return root() / "config" / Path(*parts)


@lru_cache(maxsize=1)
def _configured_paths() -> dict[str, Any]:
    paths_file = config("paths.yaml")
    with paths_file.open(encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{paths_file} did not parse to a YAML mapping")
    return loaded


def path(name: str) -> Path:
    try:
        value = _configured_paths()[name]
    except KeyError as exc:
        raise KeyError(f"missing path key in config/paths.yaml: {name}") from exc
    return root() / Path(str(value))
