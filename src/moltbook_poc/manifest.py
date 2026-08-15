from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "manifest.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save(run_dir: Path, updates: Dict[str, Any]) -> None:
    data = load(run_dir)
    _deep_update(data, updates)
    (run_dir / "manifest.json").write_text(
        json.dumps(data, indent=2, default=str), encoding="utf-8"
    )


def _deep_update(base: dict, updates: dict) -> None:
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def software_versions() -> Dict[str, Any]:
    result: Dict[str, Any] = {"python": sys.version}
    try:
        raw = subprocess.check_output(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            stderr=subprocess.DEVNULL,
        )
        pkgs = json.loads(raw)
        result["packages"] = {p["name"]: p["version"] for p in pkgs}
    except Exception:
        pass
    return result


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
