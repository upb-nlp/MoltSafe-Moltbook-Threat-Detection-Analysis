
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
DEFAULT_BASE = repo_paths.path("judge_dir") / "results.csv"
DEFAULT_RETRY = repo_paths.path("judge_dir") / "retry_results.csv"
DEFAULT_OUTPUT = repo_paths.path("judge_results")
IDENTITY_COLS = ["example_num", "node_id", "node_type"]


def read_results(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE,
                    help=f"Original run results.csv (default: {DEFAULT_BASE}).")
    ap.add_argument("--retry", type=Path, default=DEFAULT_RETRY,
                    help=f"Retry run results.csv (default: {DEFAULT_RETRY}).")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                    help=f"Where to write the merged results (default: {DEFAULT_OUTPUT}).")
    args = ap.parse_args()

    for label, path in (("base", args.base), ("retry", args.retry)):
        if not path.exists():
            sys.exit(f"ERROR: {label} results.csv not found: {path}")

    base = read_results(args.base)
    retry = read_results(args.retry)
    if list(base.columns) != list(retry.columns):
        sys.exit(f"ERROR: base and retry columns differ.\n  base : {list(base.columns)}\n"
                 f"  retry: {list(retry.columns)}")

    missing = set(retry["node_id"]) - set(base["node_id"])
    if missing:
        sys.exit(f"ERROR: {len(missing)} retry node_id(s) are not in the base file, "
                 f"e.g. {list(missing)[:3]}")

    columns = list(base.columns)
    result_cols = [c for c in columns if c not in IDENTITY_COLS]

    n_base = len(base)
    before = base["verdict"].value_counts()

    base = base.set_index("node_id")
    retry = retry.set_index("node_id")
    base.loc[retry.index, result_cols] = retry[result_cols]
    merged = base.reset_index()[columns]

    assert len(merged) == n_base, "row count changed during merge"
    assert not merged["node_id"].duplicated().any(), "duplicate node_id after merge"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)

    after = merged["verdict"].value_counts()
    print(f"replaced {len(retry)} rows -> {args.output}")
    print(f"  before: {before.to_dict()}")
    print(f"  after : {after.to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
