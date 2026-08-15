from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from openai import APIError, APITimeoutError, RateLimitError  


import pandas as pd

from moltbook_poc import repo_paths

MODEL = "gpt-5.5"                 
REASONING_EFFORT = "high"         
MAX_OUTPUT_TOKENS = 4096     
JSON_MODE = True                  

REQUESTS_PER_MINUTE = 30         
MAX_RETRIES = 5                   
REQUEST_TIMEOUT_SEC = 300       


PROMPT_CACHE_KEY = "moltbook-node-eval" 
PROMPT_CACHE_RETENTION = "24h"
CACHE_MIN_TOKENS = 1024          

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = repo_paths.root()
DEFAULT_INPUT_CSV = repo_paths.path("judge_sample_input")
DEFAULT_PROMPT = _HERE / "prompt.txt"
DEFAULT_OUTPUT_DIR = repo_paths.path("judge_dir")

RESULT_COLS = [
    "example_num", "node_id", "node_type",
    "verdict", "severity", "risk_taxonomies", "owasp_risk_codes", "harmful_intent",
    "cached_tokens", "input_tokens", "output_tokens", "reasoning_tokens",
    "model", "reasoning_effort", "ok",
]

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {path}\n"
            "Create it and write your evaluation instructions there."
        )
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt file is empty: {path}")
    return text


def _field(row: pd.Series, key: str) -> str:
    value = row.get(key, "")
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def render_node(row: pd.Series) -> str:

    node_type = _field(row, "node_type")
    title = _field(row, "title")
    text = _field(row, "text")

    lines = [f"node_type: {node_type or 'unknown'}"]
    if title:
        lines.append(f"title: {title}")
    lines.append("text:")
    lines.append(text)
    if _field(row, "is_truncated").lower() == "true":
        lines.append("\n[note: this text was truncated during sampling]")
    return "\n".join(lines)


def wrap_sample(node_text: str) -> str:

    safe = node_text.replace("</sample>", "<⁄sample>")  
    return f"<sample>\n{safe}\n</sample>"


def build_input(prompt_text: str, node_text: str) -> List[Dict[str, str]]:

    return [
        {"role": "user", "content": prompt_text},
        {"role": "user", "content": wrap_sample(node_text)},
    ]



def build_request_body(prompt_text: str, node_text: str,
                       model: str, effort: str) -> Dict[str, Any]:

    body: Dict[str, Any] = {
        "model": model,
        "reasoning": {"effort": effort},
        "input": build_input(prompt_text, node_text),
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }
    if JSON_MODE:
        body["text"] = {"format": {"type": "json_object"}}
    return body


def call_openai(client: Any, prompt_text: str, node_text: str,
                model: str, effort: str) -> Any:

    kwargs: Dict[str, Any] = build_request_body(prompt_text, node_text, model, effort)

    kwargs["prompt_cache_key"] = PROMPT_CACHE_KEY
    kwargs["prompt_cache_retention"] = PROMPT_CACHE_RETENTION
    kwargs["timeout"] = REQUEST_TIMEOUT_SEC

    last_exc: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            return client.responses.create(**kwargs)
        except (RateLimitError, APITimeoutError, APIError) as exc:
            last_exc = exc
            wait = float(2 ** (attempt + 1)) 
            print(f"    {type(exc).__name__}; retry {attempt + 1}/{MAX_RETRIES} in {wait:.0f}s")
            time.sleep(wait)

    assert last_exc is not None
    raise last_exc


def _join_labels(value: Any) -> str:

    if isinstance(value, list):
        return "; ".join(str(v).strip() for v in value if str(v).strip())
    if value is None:
        return ""
    return str(value).strip()


def parse_verdict(content: str) -> Dict[str, Any]:

    text = (content or "").strip()
    obj: Optional[Dict[str, Any]] = None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                obj = json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                obj = None

    if not isinstance(obj, dict):
        return {
            "verdict": "unparseable",
            "severity": "",
            "risk_taxonomies": "",
            "owasp_risk_codes": "",
            "harmful_intent": text[:500],
        }

    verdict = obj.get("verdict", obj.get("label", ""))

    severity: Any = obj.get("severity", "")
    try:
        severity = int(severity)
    except (TypeError, ValueError):
        severity = ""

    taxonomies = obj.get("risk_taxonomy_categories", obj.get("risk_taxonomies"))
    return {
        "verdict": str(verdict)[:100],
        "severity": severity,
        "risk_taxonomies": _join_labels(taxonomies),
        "owasp_risk_codes": _join_labels(obj.get("OWASP_risk_codes")),
        "harmful_intent": str(obj.get("harmful_intent", ""))[:500],
    }


def usage_counts(response: Any) -> Dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"input_tokens": "", "output_tokens": "", "cached_tokens": "", "reasoning_tokens": ""}

    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    return {
        "input_tokens": getattr(usage, "input_tokens", ""),
        "output_tokens": getattr(usage, "output_tokens", ""),
        "cached_tokens": getattr(input_details, "cached_tokens", "") if input_details else "",
        "reasoning_tokens": getattr(output_details, "reasoning_tokens", "") if output_details else "",
    }


def load_done_node_ids(results_csv: Path) -> Set[str]:
    if not results_csv.exists():
        return set()
    done = pd.read_csv(results_csv, usecols=["node_id"])
    return set(done["node_id"].astype(str))


def append_result_row(results_csv: Path, row: Dict[str, Any]) -> None:
    new_file = not results_csv.exists()
    with results_csv.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_COLS)
        if new_file:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in RESULT_COLS})


def append_raw(raw_jsonl: Path, obj: Dict[str, Any]) -> None:
    with raw_jsonl.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


def write_manifest(manifest_path: Path, manifest: Dict[str, Any]) -> None:
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8") 
    except Exception:
        pass

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT_CSV, help="Sample-index CSV to read.")
    ap.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT, help="Prompt file (your instructions).")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Where to write results.")
    ap.add_argument("--limit", type=int, default=0, help="Process at most N not-yet-done nodes (0 = all).")
    ap.add_argument("--model", default=MODEL, help=f"Model id (default: {MODEL}).")
    ap.add_argument("--effort", default=REASONING_EFFORT, help=f"Reasoning effort (default: {REASONING_EFFORT}).")
    ap.add_argument("--dry-run", action="store_true", help="Print the first request and exit; no API call.")
    args = ap.parse_args()

    if not args.input.exists():
        print(f"ERROR: input CSV not found: {args.input}", file=sys.stderr)
        return 1

    prompt_text = load_prompt(args.prompt)
    approx_tokens = len(prompt_text) // 4  

    sample = pd.read_csv(args.input).sort_values("example_num").reset_index(drop=True)
    if args.dry_run:
        first = sample.iloc[0]
        print("=" * 80)
        print(f"MODEL: {args.model} | reasoning effort: {args.effort} | JSON_MODE: {JSON_MODE}")
        print(f"prompt: {args.prompt}  (~{approx_tokens} tokens)")
        if approx_tokens < CACHE_MIN_TOKENS:
            print(f"note: prompt is under ~{CACHE_MIN_TOKENS} tokens, so prompt caching may not engage.")
        print("=" * 80)
        print("--- MESSAGE 1 (static prompt prefix) ---")
        print(prompt_text)
        print("\n--- MESSAGE 2 (node to evaluate, wrapped) ---")
        print(wrap_sample(render_node(first)))
        print("=" * 80)
        print("Dry run only — no API call made.")
        return 0

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("ERROR: set OPENAI_API_KEY in your environment first.", file=sys.stderr)
        print('  PowerShell:  $env:OPENAI_API_KEY = "sk-..."', file=sys.stderr)
        return 2

    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: the `openai` package is not installed.", file=sys.stderr)
        print("  .venv/Scripts/python.exe -m pip install openai", file=sys.stderr)
        return 2

    client = OpenAI(api_key=api_key)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_csv = args.output_dir / "results.csv"
    raw_jsonl = args.output_dir / "raw_responses.jsonl"
    manifest_path = args.output_dir / "manifest.json"

    done = load_done_node_ids(results_csv)
    todo = sample[~sample["node_id"].astype(str).isin(done)]
    if args.limit and args.limit > 0:
        todo = todo.head(args.limit)

    print(f"sample: {len(sample)} | already done: {len(done)} | to process now: {len(todo)}")
    if approx_tokens < CACHE_MIN_TOKENS:
        print(f"note: prompt ~{approx_tokens} tokens < {CACHE_MIN_TOKENS}; prompt caching may not engage.")
    if todo.empty:
        print("nothing to do — all selected nodes already have results.")
        return 0

    sleep_between = 60.0 / max(1, REQUESTS_PER_MINUTE)
    started_at = utc_now_iso()
    processed = 0
    verdict_tally: Dict[str, int] = {}
    cached_total = 0
    stop_reason = "completed"

    try:
        for _, node in todo.iterrows():
            example_num = int(node["example_num"])
            node_id = str(node["node_id"])
            print(f"  EXAMPLE {example_num:03d} | {node.get('node_type', ''):7} {node_id[:16]} ...", end=" ")

            try:
                response = call_openai(client, prompt_text, render_node(node),
                                       args.model, args.effort)
            except Exception as exc:  
                print(f"-> FAILED ({type(exc).__name__}), skipped (retry on re-run)")
                time.sleep(sleep_between)
                continue

            content = getattr(response, "output_text", "") or ""
            verdict = parse_verdict(content)
            usage = usage_counts(response)
            try:
                cached_total += int(usage.get("cached_tokens") or 0)
            except (TypeError, ValueError):
                pass

            append_result_row(results_csv, {
                "example_num": example_num,
                "node_id": node_id,
                "node_type": node.get("node_type", ""),
                "model": args.model,
                "reasoning_effort": args.effort,
                "ok": True,
                **verdict,
                **usage,
            })
            append_raw(raw_jsonl, {
                "example_num": example_num,
                "node_id": node_id,
                "model": args.model,
                "reasoning_effort": args.effort,
                "output_text": content,
                "usage": usage,
            })

            verdict_tally[verdict["verdict"]] = verdict_tally.get(verdict["verdict"], 0) + 1
            processed += 1
            print(f"-> {verdict['verdict']} (cached {usage.get('cached_tokens')})")
            time.sleep(sleep_between)

    except KeyboardInterrupt:
        stop_reason = "interrupted"
        print("\nInterrupted by user. Progress is saved; re-run to resume.")

    total_done_now = len(done) + processed
    write_manifest(manifest_path, {
        "stage": "openai_node_eval",
        "model": args.model,
        "reasoning_effort": args.effort,
        "json_mode": JSON_MODE,
        "prompt_file": str(args.prompt),
        "prompt_cache_retention": PROMPT_CACHE_RETENTION,
        "started_at": started_at,
        "finished_at": utc_now_iso(),
        "stop_reason": stop_reason,
        "counts": {
            "sample_total": len(sample),
            "processed_this_run": processed,
            "done_total": total_done_now,
            "remaining": len(sample) - total_done_now,
        },
        "verdicts_this_run": verdict_tally,
        "cached_tokens_this_run": cached_total,
        "inputs": {
            "input_csv": str(args.input),
        },
        "outputs": {
            "results_csv": str(results_csv),
            "raw_responses_jsonl": str(raw_jsonl),
        },
    })

    print(f"\nprocessed {processed} this run | {total_done_now}/{len(sample)} done total")
    print(f"cached tokens this run: {cached_total}")
    print(f"results  -> {results_csv}")
    print(f"manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
