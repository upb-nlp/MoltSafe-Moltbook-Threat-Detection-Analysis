from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
import os
from typing import Any


import numpy as np
import pandas as pd

from moltbook_poc import paths, repo_paths

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = repo_paths.root()
DEFAULT_RESULTS = repo_paths.path("judge_results")
DEFAULT_CORPUS = repo_paths.path("synced_nodes")
DEFAULT_OUT_DIR = repo_paths.path("fold_data_dir")
LEGACY_RUNS_DIR = repo_paths.path("data_dir")
LEGACY_RUN_ID = "seeds"

DEFAULT_SPLIT_SEED = 20260720
DEFAULT_TEST_FRACTION = 0.30
DEFAULT_FOLD_SEED = 42
DEFAULT_FOLDS = 5
SEVERITY_LEVELS = {"0", "3", "4", "5"}


def assign_split(seeds: pd.DataFrame, test_fraction: float, split_seed: int) -> pd.Series:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    if "severity" not in seeds.columns:
        raise ValueError("seeds frame has no 'severity' column to stratify on")

    test_rows = (seeds.groupby("severity", group_keys=False)
                      .sample(frac=test_fraction, random_state=split_seed))
    split = pd.Series("train", index=seeds.index, name="split")
    split.loc[test_rows.index] = "test"
    return split


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def assign_folds(train: pd.DataFrame, folds: int, seed: int) -> pd.DataFrame:
    if folds < 2:
        fail("--folds must be at least 2")
    unexpected = set(train["severity"].astype(str)) - SEVERITY_LEVELS
    if unexpected:
        fail(f"train_seeds.parquet has unexpected severities: {sorted(unexpected)}")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for severity in sorted(SEVERITY_LEVELS):
        ids = np.array(sorted(train.loc[train["severity"] == severity, "node_id"].astype(str)))
        shuffled = rng.permutation(ids)
        for i, node_id in enumerate(shuffled):
            rows.append({"node_id": str(node_id), "severity": severity, "fold": int(i % folds)})

    assignment = pd.DataFrame(rows).sort_values("node_id").reset_index(drop=True)
    check_fold_assignment(train, assignment, folds)
    return assignment


def check_fold_assignment(train: pd.DataFrame, assignment: pd.DataFrame, folds: int) -> None:
    if len(assignment) != len(train):
        fail(f"fold assignment has {len(assignment)} rows, expected {len(train)}")
    if assignment["node_id"].duplicated().any():
        fail("fold assignment has duplicate node_id values")
    if set(assignment["node_id"]) != set(train["node_id"].astype(str)):
        fail("fold assignment does not exactly match train_seeds node_id values")
    if set(assignment["fold"]) != set(range(folds)):
        fail(f"fold assignment must use folds 0..{folds - 1}")

    merged = assignment.merge(
        train[["node_id", "severity"]].astype(str),
        on="node_id",
        how="left",
        suffixes=("_assignment", "_train"),
        validate="one_to_one",
    )
    if not (merged["severity_assignment"] == merged["severity_train"]).all():
        fail("fold assignment severity values do not match train_seeds")

    for fold in range(folds):
        fold_rows = assignment[assignment["fold"] == fold]
        if fold_rows.empty:
            fail(f"fold {fold} is empty")
        if not (fold_rows["severity"] == "5").any():
            fail(f"fold {fold} has no severity-5 node")


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.runs_dir is not None or args.run_id is not None:
        runs_dir = args.runs_dir or LEGACY_RUNS_DIR
        run_id = args.run_id or LEGACY_RUN_ID
        return paths.phase1_dir(runs_dir / run_id, create=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    return args.out_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help="Directory that receives train/test seeds and dictionary artifacts")
    ap.add_argument("--run-id", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS,
                    help="Merged judge results.csv to seed from")
    ap.add_argument("--severity", type=int, nargs="+", required=True,
                    help="Severities to include, e.g. --severity 0 4 5 (0=benign, 4/5=malicious)")
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS,
                    help="English corpus parquet (source of attack_text + the scan corpus)")
    ap.add_argument("--runs-dir", type=Path, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite an existing threat_dictionary.parquet for this run")
    ap.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION,
                    help="Fraction of each severity stratum held out of the prototypes")
    ap.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED,
                    help="RNG seed for the train/test draw; independent of the pipeline seed")
    ap.add_argument("--fold-seed", type=int, default=DEFAULT_FOLD_SEED,
                    help="RNG seed for the deterministic k-fold assignment")
    ap.add_argument("--folds", type=int, default=DEFAULT_FOLDS,
                    help="Number of deterministic folds to assign over train seeds")
    args = ap.parse_args()

    if not args.results.exists():
        print(f"ERROR: missing results file: {args.results}", file=sys.stderr)
        return 1
    if not args.corpus.exists():
        print(f"ERROR: missing corpus parquet: {args.corpus}", file=sys.stderr)
        return 1

    keep = {str(s) for s in args.severity}
    res = pd.read_csv(args.results, dtype=str, keep_default_na=False)
    seeds = res[res["severity"].isin(keep)].copy()
    if len(seeds) == 0:
        print(f"ERROR: no nodes at severities {sorted(args.severity)}", file=sys.stderr)
        return 1

    corpus = pd.read_parquet(args.corpus, columns=["node_id", "embed_text"])
    text_by_id = corpus.set_index("node_id")["embed_text"]
    missing = seeds.loc[~seeds["node_id"].isin(text_by_id.index), "node_id"]
    if len(missing):
        print(f"ERROR: {len(missing)} seed node_id(s) not in corpus, e.g. {missing.head(3).tolist()}",
              file=sys.stderr)
        return 1

    split = assign_split(seeds, args.test_fraction, args.split_seed)
    out_dir = resolve_output_dir(args)
    threat_path = out_dir / "threat_dictionary.parquet"
    fold_assignment_path = out_dir / "fold_assignment.csv"
    guarded_outputs = [threat_path, fold_assignment_path]
    existing_outputs = [path for path in guarded_outputs if path.exists()]
    if existing_outputs and not args.overwrite:
        print(
            f"ERROR: {existing_outputs[0]} exists; pass --overwrite to replace",
            file=sys.stderr,
        )
        return 1

    texts = seeds["node_id"].map(text_by_id).fillna("").astype(str)
    threat = pd.DataFrame({
        "threat_id": [f"seed:{nid}" for nid in seeds["node_id"]],
        "source": "moltbook_judge10k",
        "source_row_id": seeds["node_id"].astype(str).values,
        "attack_text": texts.values,
        "attack_text_sha256": [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts],
        "category": seeds["severity"].values,
        "severity": seeds["severity"].values,
        "split": split.values,
    })

    threat.to_parquet(threat_path, index=False)

    seed_ids_path = out_dir / "seed_node_ids.txt"
    seed_ids_path.write_text("\n".join(seeds["node_id"].astype(str).tolist()), encoding="utf-8")

    sidecar_cols = ["source_row_id", "severity", "attack_text", "threat_id"]
    for name in ("train", "test"):
        side = (threat.loc[threat["split"] == name, sidecar_cols]
                      .rename(columns={"source_row_id": "node_id"}))
        side.to_parquet(out_dir / f"{name}_seeds.parquet", index=False)
        if name == "train":
            train_seeds = side


    train_ids_path = out_dir / "train_node_ids.txt"
    train_ids_path.write_text(
        "\n".join(threat.loc[threat["split"] == "train", "source_row_id"].tolist()), encoding="utf-8")

    fold_assignment = assign_folds(train_seeds, args.folds, args.fold_seed)
    fold_assignment.to_csv(fold_assignment_path, index=False)

    (out_dir / "split_manifest.json").write_text(json.dumps({
        "test_fraction": args.test_fraction,
        "split_seed": args.split_seed,
        "fold_seed": args.fold_seed,
        "folds": args.folds,
        "severities": sorted(args.severity),
        "n_train": int((threat["split"] == "train").sum()),
        "n_test": int((threat["split"] == "test").sum()),
        "per_severity_test": (threat[threat["split"] == "test"]["severity"]
                              .value_counts().sort_index().to_dict()),
    }, indent=2), encoding="utf-8")

    dst_nodes = out_dir / "moltbook_nodes.parquet"
    if dst_nodes.exists():
        print(f"NOTE: corpus already present at {dst_nodes}; leaving as-is")
    else:
        try:
            os.link(args.corpus, dst_nodes)
            print(f"Hardlinked corpus {args.corpus} -> {dst_nodes} (no extra disk)")
        except OSError as exc:
            print(f"Hardlink failed ({exc}); copying instead ...")
            shutil.copy2(args.corpus, dst_nodes)
            print(f"Copied corpus -> {dst_nodes}")

    n_train = int((threat["split"] == "train").sum())
    n_test = int((threat["split"] == "test").sum())
    print(f"Done. {len(threat)} seeds at severities {sorted(args.severity)} -> {threat_path}")
    print(f"Severity mix: {seeds['severity'].value_counts().sort_index().to_dict()}")
    print(f"Split (test_fraction={args.test_fraction}, split_seed={args.split_seed}): "
          f"train={n_train}, test={n_test}")
    print("Per-severity test counts: "
          f"{threat[threat['split'] == 'test']['severity'].value_counts().sort_index().to_dict()}")
    print(f"Seed node_ids -> {seed_ids_path}")
    print(f"Train node_ids -> {train_ids_path}")
    print(f"Fold assignment -> {fold_assignment_path}")
    print(f"Split sidecars -> {out_dir / 'train_seeds.parquet'}, {out_dir / 'test_seeds.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
