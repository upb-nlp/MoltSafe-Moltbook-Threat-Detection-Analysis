from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from moltbook_poc import repo_paths

try:
    sys.stdout.reconfigure(encoding="utf-8")  
except Exception:
    pass

_HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = repo_paths.path("judge_results")
DEFAULT_INPUT = repo_paths.path("judge_sample_input")
DEFAULT_OUTPUT = repo_paths.path("judge_dir") / "severity_samples.csv"
OUTPUT_COLS = ["severity", "example_num", "node_id", "node_type", "verdict",
               "risk_taxonomies", "owasp_risk_codes", "harmful_intent", "title", "text"]


def sample_per_severity(nodes: pd.DataFrame, severities: list[int], n: int, seed: int) -> pd.DataFrame:

    rng = np.random.default_rng(seed)
    wanted = nodes[nodes["severity"].isin([str(s) for s in severities])]
    picks = [group.sample(n=min(n, len(group)), random_state=rng)
             for _, group in wanted.groupby("severity", sort=True)]
    return pd.concat(picks, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS,
                    help=f"Judged results.csv to sample from (default: {DEFAULT_RESULTS}).")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                    help=f"Input CSV fed to the judge, for title/text (default: {DEFAULT_INPUT}).")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                    help=f"Where to write the samples (default: {DEFAULT_OUTPUT}).")
    ap.add_argument("--n", type=int, default=15, help="How many to sample per severity (default: 15).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    ap.add_argument("--severities", type=int, nargs="+", default=[0,1,2,3,4,5],
                    help="Severity levels to sample (default: 0 1 2 3 4 5).")
    args = ap.parse_args()

    for label, path in (("results", args.results), ("input", args.input)):
        if not path.exists():
            sys.exit(f"ERROR: {label} CSV not found: {path}")

    results = pd.read_csv(args.results, dtype=str, keep_default_na=False)
    content = pd.read_csv(args.input, usecols=["node_id", "title", "text"],
                          dtype=str, keep_default_na=False)
    nodes = results.merge(content, on="node_id", how="left")

    samples = sample_per_severity(nodes, args.severities, args.n, args.seed)
    samples = samples[[c for c in OUTPUT_COLS if c in samples.columns]]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    samples.to_csv(args.output, index=False)

    print(f"wrote {len(samples)} rows -> {args.output}")
    print("sampled per severity:")
    print(samples["severity"].value_counts().sort_index().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
