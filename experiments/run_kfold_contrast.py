from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from moltbook_poc import repo_paths

EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from eval_unified import ( 
    CANONICAL_TEST_FINGERPRINT,
    POSITIVE_SEVERITIES,
    compute_metrics,
    compute_ranking_metrics,
    test_set_fingerprint,
)

PROJECT_ROOT = repo_paths.root()
DATA_DIR = repo_paths.path("fold_data_dir")
DEFAULT_CLI = [sys.executable, "-m", "moltbook_poc.cli"]
DEFAULT_OUT_ROOT = repo_paths.path("results_dir")

EXPECTED_TRAIN_ROWS = 6077
EXPECTED_TRAIN_POSITIVES = 639
EXPECTED_TEST_ROWS = 2603
EXPECTED_TEST_POSITIVES = 273
SEVERITY_LEVELS = {"0", "3", "4", "5"}

SCAN_NODE_COLUMNS = [
    "node_id",
    "node_type",
    "upvotes",
    "downvotes",
    "score",
    "depth",
    "reply_count",
    "submolt_name",
    "created_at",
    "embed_text",
]


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def positive_mask(severity: pd.Series) -> pd.Series:
    return severity.astype(str).isin(POSITIVE_SEVERITIES)


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        fail(f"{path} did not parse to a YAML mapping")
    return cfg


def contrast_severity_sets(cfg: dict) -> tuple[set[str], set[str]]:
    search = cfg.get("search", {})
    if search.get("mode") != "contrast":
        fail("this experiment requires search.mode == 'contrast'")

    malicious = {str(s) for s in search.get("malicious_severities", [])}
    benign = {str(s) for s in search.get("benign_severities", [])}
    if not malicious:
        fail("search.malicious_severities must be non-empty")
    if not benign:
        fail(
            "search.benign_severities must be non-empty; random Moltbook benign sampling "
            "is forbidden for this experiment"
        )
    return malicious, benign


def load_inputs(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(data_dir / "train_seeds.parquet")
    test = pd.read_parquet(data_dir / "test_seeds.parquet")
    dictionary = pd.read_parquet(data_dir / "threat_dictionary.parquet")

    for name, df in [("train_seeds", train), ("test_seeds", test)]:
        missing = {"node_id", "severity", "attack_text", "threat_id"} - set(df.columns)
        if missing:
            fail(f"{name}.parquet is missing columns: {sorted(missing)}")
        if df["node_id"].duplicated().any():
            fail(f"{name}.parquet has duplicate node_id values")
        if df["threat_id"].duplicated().any():
            fail(f"{name}.parquet has duplicate threat_id values")
        df["severity"] = df["severity"].astype(str)

    missing_dict = {"threat_id", "severity", "split"} - set(dictionary.columns)
    if missing_dict:
        fail(f"threat_dictionary.parquet is missing columns: {sorted(missing_dict)}")
    if dictionary["threat_id"].duplicated().any():
        fail("threat_dictionary.parquet has duplicate threat_id values")
    dictionary["severity"] = dictionary["severity"].astype(str)
    dictionary["split"] = dictionary["split"].astype(str)

    check_seed_counts(train, test)
    check_test_fingerprint(test)
    return train, test, dictionary


def check_seed_counts(train: pd.DataFrame, test: pd.DataFrame) -> None:
    train_pos = int(positive_mask(train["severity"]).sum())
    test_pos = int(positive_mask(test["severity"]).sum())
    if len(train) != EXPECTED_TRAIN_ROWS or train_pos != EXPECTED_TRAIN_POSITIVES:
        fail(
            f"train set expected {EXPECTED_TRAIN_ROWS} rows / {EXPECTED_TRAIN_POSITIVES} "
            f"positives, got {len(train)} rows / {train_pos} positives"
        )
    if len(test) != EXPECTED_TEST_ROWS or test_pos != EXPECTED_TEST_POSITIVES:
        fail(
            f"test set expected {EXPECTED_TEST_ROWS} rows / {EXPECTED_TEST_POSITIVES} "
            f"positives, got {len(test)} rows / {test_pos} positives"
        )


def check_test_fingerprint(test: pd.DataFrame) -> None:
    got = test_set_fingerprint(test["node_id"])
    if got != CANONICAL_TEST_FINGERPRINT:
        fail(
            "data/fold_data test_seeds.parquet does not match the canonical node_id set\n"
            f"  got:      {got}\n  expected: {CANONICAL_TEST_FINGERPRINT}"
        )


def assign_folds(train: pd.DataFrame, folds: int, seed: int, preview: int = 0) -> pd.DataFrame:
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
        if preview:
            for preview_fold in range(folds):
                fold_preview = shuffled[preview_fold::folds]
                shown = ", ".join(map(str, fold_preview[:preview]))
                print(
                    f"Shuffled severity {severity} fold_{preview_fold} "
                    f"first {min(preview, len(fold_preview))} node_ids: {shown}"
                )
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


def read_or_write_fold_assignment(
    path: Path,
    computed: pd.DataFrame,
    overwrite_mismatch: bool = False,
) -> pd.DataFrame:
    if path.exists():
        existing = pd.read_csv(path)
        missing = {"node_id", "severity", "fold"} - set(existing.columns)
        if missing:
            fail(f"{path} is missing fold assignment columns: {sorted(missing)}")
        existing["node_id"] = existing["node_id"].astype(str)
        existing["severity"] = existing["severity"].astype(str)
        existing["fold"] = existing["fold"].astype(int)
        existing = existing.sort_values("node_id").reset_index(drop=True)
        computed_norm = computed.sort_values("node_id").reset_index(drop=True)
        if existing.equals(computed_norm):
            return existing
        if not overwrite_mismatch:
            fail(f"existing fold assignment differs from the deterministic assignment: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    computed.to_csv(path, index=False)
    return computed


def read_fold_assignment(path: Path, computed: pd.DataFrame) -> pd.DataFrame:
    if not path.exists():
        fail(f"fold assignment not found: {path}; run prepare-contrast-data first")
    return read_or_write_fold_assignment(path, computed, overwrite_mismatch=False)


def prepare_scan_nodes(
    data_dir: Path,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    seeds = pd.concat([train, test], ignore_index=True)
    nodes = pd.read_parquet(data_dir / "moltbook_nodes.parquet", columns=SCAN_NODE_COLUMNS)
    nodes["node_id"] = nodes["node_id"].astype(str)

    ordered_ids = pd.DataFrame({"node_id": seeds["node_id"].astype(str), "_order": range(len(seeds))})
    scan_nodes = ordered_ids.merge(nodes, on="node_id", how="left", validate="one_to_one")
    missing = scan_nodes["embed_text"].isna()
    if missing.any():
        fail(f"moltbook_nodes.parquet is missing {int(missing.sum())} train/test seed nodes")

    check_scan_text_matches_seeds(scan_nodes, seeds)
    scan_nodes = scan_nodes.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    if len(scan_nodes) != len(train) + len(test):
        fail("seed-only scan input does not have train + test row count")
    return scan_nodes


def check_scan_text_matches_seeds(scan_nodes: pd.DataFrame, seeds: pd.DataFrame) -> None:
    left = seeds[["node_id", "attack_text"]].copy()
    left["node_id"] = left["node_id"].astype(str)
    check = left.merge(scan_nodes[["node_id", "embed_text"]], on="node_id", how="left")
    same_text = check["attack_text"].fillna("").astype(str) == check["embed_text"].fillna("").astype(str)
    if not same_text.all():
        bad = check.loc[~same_text, "node_id"].head(5).tolist()
        fail(
            "seed attack_text is not identical to production embed_text; refusing to "
            f"change the scoring text path. Example node_id values: {bad}"
        )


def filter_reference_dictionary(
    dictionary: pd.DataFrame,
    held_in_train: pd.DataFrame,
    malicious_severities: set[str],
    benign_severities: set[str],
) -> pd.DataFrame:
    held_ids = set(held_in_train["threat_id"].astype(str))
    train_dict = dictionary[dictionary["split"] == "train"].copy()
    filtered = train_dict[train_dict["threat_id"].astype(str).isin(held_ids)].copy()

    if len(filtered) != len(held_ids):
        found = set(filtered["threat_id"].astype(str))
        fail(f"filtered dictionary is missing {len(held_ids - found)} held-in train threat_id values")
    if len(filtered) != len(held_in_train):
        fail("filtered dictionary does not contain exactly four folds of train seeds")

    severities = filtered["severity"].astype(str)
    n_malicious = int(severities.isin(malicious_severities).sum())
    n_benign = int(severities.isin(benign_severities).sum())
    if n_malicious == 0:
        fail(f"filtered dictionary has no malicious rows for severities {sorted(malicious_severities)}")
    if n_benign == 0:
        fail(
            f"filtered dictionary has no benign rows for severities {sorted(benign_severities)}; "
            "random Moltbook benign sampling is forbidden"
        )
    return filtered


def choose_threshold(scores: np.ndarray, target_recall: float) -> tuple[float, int]:
    if len(scores) == 0:
        fail("held-out fold has no positive scores to set the threshold")
    if not 0.0 < target_recall <= 1.0:
        fail("--target-recall must be in (0, 1]")
    sorted_scores = np.sort(scores)
    wanted = int(np.ceil(len(sorted_scores) * target_recall))
    cutoff_index = len(sorted_scores) - wanted
    return float(sorted_scores[cutoff_index]), cutoff_index


def metric_block(labels: pd.Series, scores: pd.Series, flags: pd.Series) -> dict:
    y = labels.astype(int).to_numpy()
    pred = flags.astype(int).to_numpy()
    score = scores.astype(float).to_numpy()
    out = compute_metrics(y, pred)
    out.update(compute_ranking_metrics(y, score))
    return out


def flatten_metrics(prefix: str, metrics: dict) -> dict:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def run_command(command: list[str]) -> None:
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def write_fold_inputs(
    fold_dir: Path,
    reference: pd.DataFrame,
    held_in_train: pd.DataFrame,
    scan_nodes: pd.DataFrame,
) -> None:
    reference_dir = fold_dir / "reference"
    phase1_dir = fold_dir / "phase1"
    reference_dir.mkdir(parents=True, exist_ok=True)
    phase1_dir.mkdir(parents=True, exist_ok=True)

    reference.to_parquet(reference_dir / "threat_dictionary.parquet", index=False)
    reference.to_parquet(phase1_dir / "threat_dictionary.parquet", index=False)
    held_in_train.to_parquet(reference_dir / "train_seeds.parquet", index=False)
    (reference_dir / "train_node_ids.txt").write_text(
        "\n".join(held_in_train["node_id"].astype(str).tolist()), encoding="utf-8"
    )
    scan_nodes.to_parquet(phase1_dir / "scan_nodes.parquet", index=False)


def evaluate_fold(
    fold: int,
    fold_dir: Path,
    train: pd.DataFrame,
    test: pd.DataFrame,
    assignment: pd.DataFrame,
    target_recall: float,
) -> dict:
    scan = pd.read_parquet(fold_dir / "phase1" / "scan_results.parquet")
    if len(scan) != len(train) + len(test):
        fail(f"fold {fold}: scan covered {len(scan)} rows, expected {len(train) + len(test)}")
    if scan["node_id"].duplicated().any():
        fail(f"fold {fold}: scan_results.parquet has duplicate node_id values")
    if scan["max_similarity"].isna().any():
        fail(f"fold {fold}: scan_results.parquet contains NaN max_similarity scores")

    fold_ids = set(assignment.loc[assignment["fold"] == fold, "node_id"].astype(str))
    validation = train[train["node_id"].astype(str).isin(fold_ids)].copy()
    validation["label"] = positive_mask(validation["severity"]).astype(int)

    scan_scores = scan[["node_id", "max_similarity"]].rename(columns={"max_similarity": "score"})
    validation = validation.merge(scan_scores, on="node_id", how="left", validate="one_to_one")
    if validation["score"].isna().any():
        fail(f"fold {fold}: missing validation scores")

    positive_scores = validation.loc[validation["label"] == 1, "score"].astype(float).to_numpy()
    threshold, cutoff_index = choose_threshold(positive_scores, target_recall)

    validation["flag_tuned"] = validation["score"] >= threshold
    val_metrics = compute_metrics(
        validation["label"].astype(int).to_numpy(),
        validation["flag_tuned"].astype(int).to_numpy(),
    )
    granularity = 1.0 / int(validation["label"].sum())
    if abs(float(val_metrics["recall"]) - target_recall) > granularity:
        fail(
            f"fold {fold}: validation recall {val_metrics['recall']:.6f} is not within "
            f"one positive node ({granularity:.6f}) of target {target_recall:.6f}"
        )

    test_scored = test[["node_id", "severity"]].copy()
    test_scored["label"] = positive_mask(test_scored["severity"]).astype(int)
    test_scored = test_scored.merge(scan_scores, on="node_id", how="left", validate="one_to_one")
    if test_scored["score"].isna().any():
        fail(f"fold {fold}: missing test scores")

    test_scored["flag_tuned"] = test_scored["score"] >= threshold
    test_scored[["node_id", "score", "flag_tuned"]].to_parquet(
        fold_dir / "test_scores.parquet", index=False
    )

    tuned = metric_block(test_scored["label"], test_scored["score"], test_scored["flag_tuned"])

    threshold_info = {
        "fold": fold,
        "target_recall": target_recall,
        "threshold": threshold,
        "cutoff_index_in_sorted_positive_scores": cutoff_index,
        "validation_recall": val_metrics["recall"],
        "validation_positive_count": int(validation["label"].sum()),
        "validation_ties_at_threshold": int((validation["score"] == threshold).sum()),
        "validation_positive_ties_at_threshold": int(
            ((validation["score"] == threshold) & (validation["label"] == 1)).sum()
        ),
        "test_ties_at_threshold": int((test_scored["score"] == threshold).sum()),
        "test_positive_ties_at_threshold": int(
            ((test_scored["score"] == threshold) & (test_scored["label"] == 1)).sum()
        ),
        "flag_rule": "score >= threshold",
    }
    (fold_dir / "threshold.json").write_text(
        json.dumps(json_ready(threshold_info), indent=2), encoding="utf-8"
    )

    metrics = {
        "fold": fold,
        "threshold": threshold_info,
        "validation": val_metrics,
        "tuned": tuned,
    }
    (fold_dir / "metrics.json").write_text(
        json.dumps(json_ready(metrics), indent=2), encoding="utf-8"
    )

    row = {
        "fold": fold,
        "threshold": threshold,
        "validation_recall": val_metrics["recall"],
        "validation_positive_count": threshold_info["validation_positive_count"],
        "validation_ties_at_threshold": threshold_info["validation_ties_at_threshold"],
        "test_ties_at_threshold": threshold_info["test_ties_at_threshold"],
    }
    row.update(flatten_metrics("tuned", tuned))
    return row


def write_summary(model_dir: Path, rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    numeric_cols = [c for c in df.columns if c != "fold" and pd.api.types.is_numeric_dtype(df[c])]
    mean = {"fold": "mean", **{c: df[c].mean() for c in numeric_cols}}
    std = {"fold": "std", **{c: df[c].std(ddof=1) for c in numeric_cols}}
    out = pd.concat([df, pd.DataFrame([mean, std])], ignore_index=True)
    out.to_csv(model_dir / "summary.csv", index=False)
    return out


def print_summary(summary: pd.DataFrame) -> None:
    cols = [
        "fold",
        "tuned_average_precision",
        "tuned_roc_auc",
        "tuned_precision",
    ]
    print()
    print(summary[cols].to_string(index=False))
    mean = summary[summary["fold"] == "mean"].iloc[0]
    std = summary[summary["fold"] == "std"].iloc[0]
    print()
    print(
        "Average Precision:      "
        f"mean={mean['tuned_average_precision']:.4f}, "
        f"std={std['tuned_average_precision']:.4f}"
    )
    print(
        "ROC-AUC:                "
        f"mean={mean['tuned_roc_auc']:.4f}, std={std['tuned_roc_auc']:.4f}"
    )
    print(
        "Precision @ recall .80: "
        f"mean={mean['tuned_precision']:.4f}, std={std['tuned_precision']:.4f}"
    )
    print(
        "Forwarding rate:        "
        f"mean={mean['tuned_flag_rate']:.4f}, std={std['tuned_flag_rate']:.4f}"
    )


def guard_output_dir(model_dir: Path, folds: int, overwrite: bool) -> None:
    if not model_dir.exists():
        return
    result_names = {"summary.csv", "kfold_manifest.json"} | {f"fold_{i}" for i in range(folds)}
    existing = [p for p in model_dir.iterdir() if p.name in result_names]
    if existing and not overwrite:
        fail(f"{model_dir} already contains k-fold outputs; pass --overwrite to replace them")


def write_manifest(model_dir: Path, args: argparse.Namespace, config_path: Path) -> None:
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "data_dir": str(args.data_dir),
        "folds": args.folds,
        "fold_seed": args.seed,
        "target_recall": args.target_recall,
        "positive_severities": sorted(POSITIVE_SEVERITIES),
        "commands_use_cli": args.cli,
    }
    (model_dir / "kfold_manifest.json").write_text(
        json.dumps(json_ready(manifest), indent=2), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--model-config-name", default=None)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--cli",
        default=None,
        help="Optional Moltbook CLI executable; default uses this Python with -m moltbook_poc.cli",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--target-recall", type=float, default=0.8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.cli = [args.cli] if args.cli else list(DEFAULT_CLI)

    config_path = args.config.resolve()
    cfg = load_config(config_path)
    malicious_severities, benign_severities = contrast_severity_sets(cfg)

    train, test, dictionary = load_inputs(args.data_dir)
    computed_assignment = assign_folds(train, args.folds, args.seed)

    model_name = args.model_config_name or config_path.stem
    model_dir = args.out_root / model_name
    guard_output_dir(model_dir, args.folds, args.overwrite)
    model_dir.mkdir(parents=True, exist_ok=True)

    data_assignment = read_fold_assignment(args.data_dir / "fold_assignment.csv", computed_assignment)
    root_assignment = read_or_write_fold_assignment(
        args.out_root / "fold_assignment.csv", data_assignment, overwrite_mismatch=False
    )
    assignment = read_or_write_fold_assignment(
        model_dir / "fold_assignment.csv", root_assignment, overwrite_mismatch=args.overwrite
    )
    check_fold_assignment(train, assignment, args.folds)

    scan_nodes = prepare_scan_nodes(args.data_dir, train, test)
    print(f"Seed-only scan input: {len(scan_nodes)} nodes")
    print(f"Malicious severities: {sorted(malicious_severities)}")
    print(f"Benign severities:    {sorted(benign_severities)}")

    rows: list[dict] = []
    for fold in range(args.folds):
        print(f"\n=== fold_{fold} ===", flush=True)
        fold_dir = model_dir / f"fold_{fold}"
        held_out_ids = set(assignment.loc[assignment["fold"] == fold, "node_id"].astype(str))
        held_in_train = train[~train["node_id"].astype(str).isin(held_out_ids)].copy()
        reference = filter_reference_dictionary(
            dictionary, held_in_train, malicious_severities, benign_severities
        )
        print(
            f"  reference rows: {len(reference)} "
            f"({int(reference['severity'].isin(malicious_severities).sum())} malicious, "
            f"{int(reference['severity'].isin(benign_severities).sum())} benign)"
        )

        write_fold_inputs(fold_dir, reference, held_in_train, scan_nodes)

        run_command([
            *args.cli,
            "build-index",
            "--run-id",
            f"fold_{fold}",
            "--runs-dir",
            str(model_dir),
            "--config",
            str(config_path),
            "--overwrite",
        ])
        run_command([
            *args.cli,
            "scan-moltbook",
            "--run-id",
            f"fold_{fold}",
            "--runs-dir",
            str(model_dir),
            "--config",
            str(config_path),
            "--nodes-path",
            str(fold_dir / "phase1" / "scan_nodes.parquet"),
            "--overwrite",
        ])

        row = evaluate_fold(fold, fold_dir, train, test, assignment, args.target_recall)
        rows.append(row)
        print(
            f"  ap={row['tuned_average_precision']:.4f}, "
            f"roc={row['tuned_roc_auc']:.4f}, "
            f"precision={row['tuned_precision']:.4f}"
        )

    summary = write_summary(model_dir, rows)
    write_manifest(model_dir, args, config_path)

    shutil.copy2(args.out_root / "fold_assignment.csv", model_dir / "fold_assignment.csv")
    print_summary(summary)
    print(f"\nwrote {model_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
