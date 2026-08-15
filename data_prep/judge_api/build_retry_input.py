from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from moltbook_poc import repo_paths

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = repo_paths.path("judge_dir") / "results.csv"
DEFAULT_SOURCE = repo_paths.path("judge_sample_input")
DEFAULT_OUTPUT = repo_paths.path("judge_retry_input")
DEFAULT_VERDICTS = "error,unparseable"

OUTPUT_COLS = ["example_num", "node_id", "node_type", "title", "text", "is_truncated"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS,
                    help=f"Fetched results CSV to select from (default: {DEFAULT_RESULTS}).")
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                    help=f"Original sent input CSV, source of title/text (default: {DEFAULT_SOURCE}).")
    ap.add_argument("--verdicts", default=DEFAULT_VERDICTS,
                    help=f"Comma-separated verdicts to retry (default: {DEFAULT_VERDICTS}).")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                    help=f"Where to write the retry input CSV (default: {DEFAULT_OUTPUT}).")
    args = ap.parse_args()

    for label, path in (("results", args.results), ("source", args.source)):
        if not path.exists():
            sys.exit(f"ERROR: {label} CSV not found: {path}")

    wanted = {v.strip() for v in args.verdicts.split(",") if v.strip()}
    if not wanted:
        sys.exit("ERROR: --verdicts selected nothing; pass e.g. --verdicts error,unparseable")

    results = pd.read_csv(args.results, dtype=str)
    remaining_ids = results.loc[results["verdict"].isin(wanted), "node_id"].tolist()
    if not remaining_ids:
        sys.exit(f"ERROR: no rows in {args.results} with verdict in {sorted(wanted)}.")
    remaining_ids = list(dict.fromkeys(remaining_ids))

    source = pd.read_csv(args.source, dtype=str)
    src_by_id = source.set_index("node_id")

    missing = [nid for nid in remaining_ids if nid not in src_by_id.index]
    if missing:
        print(f"WARNING: {len(missing)} selected node_id(s) absent from {args.source} — "
              f"they cannot be retried and are dropped. First few: {missing[:3]}")
    present = [nid for nid in remaining_ids if nid in src_by_id.index]

    sample = src_by_id.loc[present].reset_index()  

    blank = sample["text"].fillna("").str.strip() == ""
    if blank.any():
        print(f"WARNING: {int(blank.sum())} selected node(s) have blank text in the source — "
              f"kept as-is (they were sent originally too).")

    sample["example_num"] = pd.to_numeric(sample["example_num"], errors="coerce")
    sample = sample.sort_values("example_num", kind="stable").reset_index(drop=True)
    sample["example_num"] = range(1, len(sample) + 1)
    if "is_truncated" not in sample.columns:
        sample["is_truncated"] = False
    out = sample[OUTPUT_COLS]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"selected verdicts {sorted(wanted)}: {len(remaining_ids)} node(s); "
          f"wrote {len(out)} rows -> {args.output}")
    print(out["node_type"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
