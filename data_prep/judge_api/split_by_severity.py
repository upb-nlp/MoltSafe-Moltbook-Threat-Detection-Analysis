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
DEFAULT_RESULTS_DIR = repo_paths.path("judge_dir")
DEFAULT_INPUT = repo_paths.path("judge_sample_input")
VERDICT_COLS = ["verdict", "severity", "risk_taxonomies", "owasp_risk_codes", "harmful_intent"]
NODES_CSV_COLS = ["example_num", "node_id", "node_type", "title", "text", *VERDICT_COLS]
DIVIDER = "\n\n" + "-" * 70 + "\n\n"


def render_node_block(row: pd.Series) -> str:
    def val(key: str) -> str:
        v = row.get(key, "")
        return "" if pd.isna(v) else str(v)

    lines = [
        f"example_num: {val('example_num')}",
        f"node_id:     {val('node_id')}",
        f"node_type:   {val('node_type')}",
        f"verdict:     {val('verdict')}  |  severity: {val('severity')}",
        f"risk_taxonomies: {val('risk_taxonomies')}",
        f"owasp_risk_codes: {val('owasp_risk_codes')}",
        f"harmful_intent:  {val('harmful_intent')}",
    ]
    if val("title"):
        lines.append(f"title: {val('title')}")
    lines.append("")
    lines.append(val("text"))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
                    help=f"Run folder holding results.csv (default: {DEFAULT_RESULTS_DIR}).")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                    help=f"Input CSV fed to the judge, for title/text (default: {DEFAULT_INPUT}).")
    ap.add_argument("--severities", type=int, nargs="+", default=[1, 2],
                    help="Severity levels to split out (default: 1 2).")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Base folder for the severity_* folders (default: the results dir).")
    args = ap.parse_args()

    results_csv = args.results_dir / "results.csv"
    if not results_csv.exists():
        sys.exit(f"ERROR: results.csv not found: {results_csv}")
    if not args.input.exists():
        sys.exit(f"ERROR: input CSV not found: {args.input}")

    results = pd.read_csv(results_csv)
    nodes = pd.read_csv(args.input, usecols=["node_id", "title", "text"])

    merged = results.merge(nodes, on="node_id", how="left")
    merged["severity"] = pd.to_numeric(merged["severity"], errors="coerce")

    out_base = args.out_dir or args.results_dir
    for sev in args.severities:
        subset = merged[merged["severity"] == sev].sort_values("example_num")
        folder = out_base / f"severity_{sev}"
        folder.mkdir(parents=True, exist_ok=True)

        cols = [c for c in NODES_CSV_COLS if c in subset.columns]
        subset[cols].to_csv(folder / "nodes.csv", index=False)

        for stale in folder.glob("example_*.txt"):
            stale.unlink()

        header = f"severity {sev} — {len(subset)} node(s)"
        blocks = [render_node_block(row) for _, row in subset.iterrows()]
        (folder / "nodes.txt").write_text(header + DIVIDER + DIVIDER.join(blocks) + "\n",
                                          encoding="utf-8")

        print(f"severity {sev}: {len(subset):3d} nodes -> {folder}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
