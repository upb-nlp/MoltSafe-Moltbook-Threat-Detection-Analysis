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
PROJECT_ROOT = repo_paths.root()
DEFAULT_PARQUET = repo_paths.path("corpus_english")
DEFAULT_OUTPUT = repo_paths.path("judge_sample_input")
OUTPUT_COLS = ["example_num", "node_id", "node_type", "title", "text", "is_truncated"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, required=True, help="How many nodes to sample.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET,
                    help=f"Source parquet (default: {DEFAULT_PARQUET}).")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                    help=f"Where to write the input CSV (default: {DEFAULT_OUTPUT}).")
    args = ap.parse_args()

    if not args.parquet.exists():
        sys.exit(f"ERROR: parquet not found: {args.parquet}")

    df = pd.read_parquet(args.parquet, columns=["node_id", "node_type", "title", "text"])

    has_text = df["text"].fillna("").str.strip() != ""
    df = df[has_text]

    if args.n > len(df):
        print(f"note: asked for {args.n} but only {len(df)} nodes have text; taking all of them.")
        sample = df
    else:
        sample = df.sample(n=args.n, random_state=args.seed)

    out = sample.reset_index(drop=True).copy()
    out.insert(0, "example_num", range(1, len(out) + 1))
    out["is_truncated"] = False
    out = out[OUTPUT_COLS]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"wrote {len(out)} rows -> {args.output}")
    print(out["node_type"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
