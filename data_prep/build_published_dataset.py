from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

from moltbook_poc import repo_paths

REPO = repo_paths.root()
DEFAULT_RESULTS = repo_paths.path("judge_results")
DEFAULT_CORPUS = repo_paths.path("corpus_english")
DEFAULT_OUTPUT = repo_paths.path("published_dataset")

OUTPUT_COLUMNS = [
    "node_id",
    "node_type",
    "verdict",
    "severity",
    "risk_taxonomies",
    "owasp_risk_codes",
    "text_sha256",
]


def text_sha256(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def build_dataset(results: Path, corpus: Path, output: Path) -> int:
    judged = pd.read_csv(results, dtype=str, keep_default_na=False)
    missing_results = sorted(set(OUTPUT_COLUMNS[:-1]) - set(judged.columns))
    if missing_results:
        raise ValueError(f"{results} missing columns: {missing_results}")
    if len(judged) != 10000:
        raise ValueError(f"{results} has {len(judged)} rows, expected 10000")
    if judged["node_id"].duplicated().any():
        raise ValueError(f"{results} contains duplicate node_id values")

    corpus_df = pd.read_parquet(corpus, columns=["node_id", "embed_text"])
    corpus_df["node_id"] = corpus_df["node_id"].astype(str)
    corpus_df["text_sha256"] = corpus_df["embed_text"].map(text_sha256)

    out = judged[OUTPUT_COLUMNS[:-1]].merge(
        corpus_df[["node_id", "text_sha256"]],
        on="node_id",
        how="left",
        validate="one_to_one",
    )
    missing_hash = out.loc[out["text_sha256"].isna(), "node_id"].tolist()
    if missing_hash:
        shown = ", ".join(missing_hash[:20])
        raise ValueError(f"{len(missing_hash)} node_id values missing from corpus: {shown}")

    output.parent.mkdir(parents=True, exist_ok=True)
    out[OUTPUT_COLUMNS].to_csv(output, index=False)
    return len(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    try:
        rows = build_dataset(args.results, args.corpus, args.output)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {rows} rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
