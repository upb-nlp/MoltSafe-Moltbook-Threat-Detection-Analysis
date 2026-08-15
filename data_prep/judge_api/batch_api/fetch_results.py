from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

import batch_common as bc
from batch_common import ne

NOT_READY = {"validating", "in_progress", "finalizing", "cancelling"}


def _read_jsonl(client: Any, file_id: str) -> List[Dict[str, Any]]:
    text = client.files.content(file_id).text
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _base_row(node_id: str, meta: Dict[str, tuple], state: Dict[str, Any]) -> Dict[str, Any]:
    example_num, node_type = meta.get(node_id, ("", ""))
    return {
        "example_num": example_num,
        "node_id": node_id,
        "node_type": node_type,
        "model": state["model"],
        "reasoning_effort": state["effort"],
    }


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, required=True, help="Dir holding batch_state.json.")
    args = ap.parse_args()

    state = bc.load_state(args.output_dir)
    client = bc.make_client()
    batch = client.batches.retrieve(state["batch_id"])

    if batch.status in NOT_READY:
        print(f"batch {batch.id} is '{batch.status}', not ready yet — re-run later.", file=sys.stderr)
        return 1
    print(f"batch {batch.id} status: {batch.status}")

    src = pd.read_csv(state["input_csv"])
    meta = {str(r["node_id"]): (int(r["example_num"]), str(r["node_type"])) for _, r in src.iterrows()}

    result_rows: List[Dict[str, Any]] = []
    raw_rows: List[Dict[str, Any]] = []
    seen: set = set()
    tally: Dict[str, int] = {}

    for line in (_read_jsonl(client, batch.output_file_id) if getattr(batch, "output_file_id", None) else []):
        node_id = str(line.get("custom_id", ""))
        seen.add(node_id)
        row = _base_row(node_id, meta, state)
        resp = line.get("response") or {}
        if resp.get("status_code") == 200 and resp.get("body"):
            parsed = bc.parse_success_body(resp["body"])
            row.update({k: v for k, v in parsed.items() if not k.startswith("_")})
            row["ok"] = True
            raw_rows.append({**_base_row(node_id, meta, state),
                             "output_text": parsed["_output_text"], "usage": parsed["_usage"]})
        else:
            row.update({"verdict": "error", "ok": False,
                        "harmful_intent": json.dumps(resp)[:500] or "non-200 response"})
            raw_rows.append({**_base_row(node_id, meta, state), "output_text": "", "usage": {},
                             "error": resp})
        tally[row.get("verdict", "")] = tally.get(row.get("verdict", ""), 0) + 1
        result_rows.append(row)

    for line in (_read_jsonl(client, batch.error_file_id) if getattr(batch, "error_file_id", None) else []):
        node_id = str(line.get("custom_id", ""))
        if node_id in seen:
            continue
        seen.add(node_id)
        err = line.get("error") or line.get("response") or {}
        row = _base_row(node_id, meta, state)
        row.update({"verdict": "error", "ok": False, "harmful_intent": json.dumps(err)[:500]})
        tally["error"] = tally.get("error", 0) + 1
        result_rows.append(row)
        raw_rows.append({**_base_row(node_id, meta, state), "output_text": "", "usage": {}, "error": err})

    for node_id in meta:
        if node_id in seen:
            continue
        row = _base_row(node_id, meta, state)
        row.update({"verdict": "missing", "ok": False, "harmful_intent": "no response in batch output"})
        tally["missing"] = tally.get("missing", 0) + 1
        result_rows.append(row)
        raw_rows.append({**_base_row(node_id, meta, state), "output_text": "", "usage": {}})

    result_rows.sort(key=lambda r: (r["example_num"] == "", r["example_num"]))
    raw_rows.sort(key=lambda r: (r["example_num"] == "", r["example_num"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_csv = args.output_dir / "results.csv"
    raw_jsonl = args.output_dir / "raw_responses.jsonl"
    manifest_path = args.output_dir / "manifest.json"

    with results_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ne.RESULT_COLS)
        writer.writeheader()
        for row in result_rows:
            writer.writerow({col: row.get(col, "") for col in ne.RESULT_COLS})

    with raw_jsonl.open("w", encoding="utf-8") as fh:
        for row in raw_rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    ne.write_manifest(manifest_path, {
        "stage": "openai_node_eval_batch",
        "model": state["model"],
        "reasoning_effort": state["effort"],
        "json_mode": ne.JSON_MODE,
        "prompt_file": state["prompt_file"],
        "batch_id": batch.id,
        "batch_status": batch.status,
        "started_at": state.get("submitted_at", ""),
        "finished_at": ne.utc_now_iso(),
        "stop_reason": batch.status,
        "counts": {
            "sample_total": len(meta),
            "written": len(result_rows),
        },
        "verdicts": tally,
        "inputs": {"input_csv": state["input_csv"]},
        "outputs": {"results_csv": str(results_csv), "raw_responses_jsonl": str(raw_jsonl)},
    })

    print(f"wrote {len(result_rows)} rows -> {results_csv}")
    print("verdicts:", tally)
    print(f"manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
