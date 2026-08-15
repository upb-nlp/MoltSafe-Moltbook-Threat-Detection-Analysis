from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))
import evaluate_nodes as ne  

BATCH_ENDPOINT = "/v1/responses"
COMPLETION_WINDOW = "24h"
STATE_FILENAME = "batch_state.json"

def make_client() -> Any:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("ERROR: set OPENAI_API_KEY in your environment first.", file=sys.stderr)
        print('  PowerShell:  $env:OPENAI_API_KEY = "sk-..."', file=sys.stderr)
        raise SystemExit(2)
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: the `openai` package is not installed.", file=sys.stderr)
        print("  .venv/Scripts/python.exe -m pip install openai", file=sys.stderr)
        raise SystemExit(2)
    return OpenAI(api_key=api_key)

def state_path(output_dir: Path) -> Path:
    return output_dir / STATE_FILENAME


def save_state(output_dir: Path, state: Dict[str, Any]) -> None:
    state_path(output_dir).write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def load_state(output_dir: Path) -> Dict[str, Any]:
    path = state_path(output_dir)
    if not path.exists():
        print(f"ERROR: no batch state at {path}. Run submit_batch.py first.", file=sys.stderr)
        raise SystemExit(1)
    return json.loads(path.read_text(encoding="utf-8"))


def to_batch_line(row: pd.Series, prompt_text: str, model: str, effort: str) -> Dict[str, Any]:

    body = ne.build_request_body(prompt_text, ne.render_node(row), model, effort)
    body["prompt_cache_key"] = ne.PROMPT_CACHE_KEY
    body["prompt_cache_retention"] = ne.PROMPT_CACHE_RETENTION
    return {
        "custom_id": str(row["node_id"]),
        "method": "POST",
        "url": BATCH_ENDPOINT,
        "body": body,
    }



def _extract_output_text(body: Dict[str, Any]) -> str:
    parts = []
    for item in body.get("output", []) or []:
        if item.get("type") == "message":
            for chunk in item.get("content", []) or []:
                if chunk.get("type") == "output_text":
                    parts.append(chunk.get("text", ""))
    return "".join(parts)


def _usage_from_body(body: Dict[str, Any]) -> Dict[str, Any]:
    usage = body.get("usage") or {}
    in_details = usage.get("input_tokens_details") or {}
    out_details = usage.get("output_tokens_details") or {}
    return {
        "input_tokens": usage.get("input_tokens", ""),
        "output_tokens": usage.get("output_tokens", ""),
        "cached_tokens": in_details.get("cached_tokens", ""),
        "reasoning_tokens": out_details.get("reasoning_tokens", ""),
    }


def parse_success_body(body: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from openai.types.responses import Response
        resp = Response.model_validate(body)
        content = resp.output_text or ""
        usage = ne.usage_counts(resp)
    except Exception:
        content = _extract_output_text(body)
        usage = _usage_from_body(body)
    return {**ne.parse_verdict(content), **usage, "_output_text": content, "_usage": usage}
