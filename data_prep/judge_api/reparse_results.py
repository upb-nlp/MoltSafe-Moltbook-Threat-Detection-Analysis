from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from moltbook_poc import repo_paths

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_nodes import parse_verdict 

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = repo_paths.path("judge_dir")
VERDICT_COLS = ["verdict", "severity", "risk_taxonomies", "owasp_risk_codes", "harmful_intent"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
                    help=f"Run folder with results.csv + raw_responses.jsonl (default: {DEFAULT_RESULTS_DIR}).")
    args = ap.parse_args()

    results_csv = args.results_dir / "results.csv"
    raw_jsonl = args.results_dir / "raw_responses.jsonl"
    if not results_csv.exists():
        sys.exit(f"ERROR: results.csv not found: {results_csv}")
    if not raw_jsonl.exists():
        sys.exit(f"ERROR: raw_responses.jsonl not found: {raw_jsonl}")

    raw_by_id: dict[str, str] = {}
    with raw_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            raw_by_id[str(obj.get("node_id"))] = obj.get("output_text", "") or ""

    df = pd.read_csv(results_csv, dtype={"node_id": str})
    df[VERDICT_COLS] = df[VERDICT_COLS].astype(object)
    changed = 0
    for i, node_id in df["node_id"].items():
        if node_id not in raw_by_id:
            continue
        parsed = parse_verdict(raw_by_id[node_id])
        before = df.loc[i, "risk_taxonomies"]
        for col in VERDICT_COLS:
            df.at[i, col] = parsed[col]
        if str(before) != str(parsed["risk_taxonomies"]):
            changed += 1

    df.to_csv(results_csv, index=False)
    print(f"re-parsed {len(df)} rows -> {results_csv}")
    print(f"risk_taxonomies changed on {changed} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
