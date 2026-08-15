from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import shutil
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import utils
from train_head import (
    DEFAULT_CONFIG,
    NUM_CLASSES,
    ON_KAGGLE,
    SCRIPT_DIR,
    Config,
    SafetyClassifier,
    load_data,
    predict_proba,
    prepare_texts,
    train_head,
)

DEFAULT_DATA_DIR = SCRIPT_DIR / "data"
if not ON_KAGGLE:
    DEFAULT_DATA_DIR = SCRIPT_DIR.parent / "data" / "fold_data"
DEFAULT_OUT_ROOT = (
    Path("/kaggle/working/head_clf_stratified/runs_kfold")
    if ON_KAGGLE
    else SCRIPT_DIR / "runs_kfold"
)

EXPECTED_TRAIN_ROWS = 6077
EXPECTED_TRAIN_POSITIVES = 639
EXPECTED_TEST_ROWS = 2603
EXPECTED_TEST_POSITIVES = 273
POSITIVE_SEVERITIES = {"3", "4", "5"}
VALID_SEVERITIES = {"0", "3", "4", "5"}
CONTRAST_SUMMARY_HEADER = [
    "fold",
    "threshold",
    "validation_recall",
    "validation_positive_count",
    "validation_ties_at_threshold",
    "test_ties_at_threshold",
    "tuned_n_flagged",
    "tuned_flag_rate",
    "tuned_tp",
    "tuned_fp",
    "tuned_fn",
    "tuned_precision",
    "tuned_recall",
    "tuned_average_precision",
    "tuned_roc_auc",
]

CONFIG_CONTRACT_FIELDS = [
    "model_name",
    "max_seq_length",
    "chunking",
    "fp16",
    "hidden",
    "dropout",
    "lr",
    "weight_decay",
    "checkpoint_every",
    "num_classes",
    "mask_emails",
    "seed",
    "variant",
    "instruct_prefix",
]
CONFIG_REPORTED_DRIFT_FIELDS = ["batch_size", "max_epochs"]


@dataclass(frozen=True)
class Inputs:
    train_texts: list[str]
    y_train: np.ndarray
    train_ids: list[str]
    test_texts: list[str]
    y_test: np.ndarray
    test_ids: list[str]
    assignment_by_row: pd.DataFrame


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


def throughput_summary(model_dir: Path, folds: int) -> dict:
    per_fold = []
    for fold in range(folds):
        metrics_path = model_dir / f"fold_{fold}" / "metrics.json"
        if not metrics_path.is_file():
            fail(f"fold {fold}: missing throughput source file: {metrics_path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        required = (
            "throughput_nodes_per_sec",
            "throughput_n_nodes",
            "throughput_seconds",
        )
        missing = [key for key in required if key not in metrics]
        if missing:
            fail(f"fold {fold}: {metrics_path} missing throughput keys: {missing}")
        per_fold.append({
            "fold": fold,
            "nodes_per_sec": float(metrics["throughput_nodes_per_sec"]),
            "n_nodes": int(metrics["throughput_n_nodes"]),
            "seconds": float(metrics["throughput_seconds"]),
        })

    rates = np.array([item["nodes_per_sec"] for item in per_fold], dtype=float)
    n_nodes = per_fold[0]["n_nodes"] if per_fold else 0
    if any(item["n_nodes"] != n_nodes for item in per_fold):
        fail("throughput_n_nodes differs across folds")
    return {
        "nodes_per_sec_mean": float(rates.mean()) if len(rates) else 0.0,
        "nodes_per_sec_std": float(rates.std(ddof=1)) if len(rates) > 1 else 0.0,
        "per_fold": per_fold,
        "n_nodes": int(n_nodes),
        "definition": "nodes / wall-clock seconds of the scored inference pass",
    }


def load_json(path: Path) -> Any:
    if not path.is_file():
        fail(f"required reference file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def check_published_config(cfg: Config, reference_path: Path) -> dict:
    expected = load_json(reference_path)
    actual = cfg.to_json()
    mismatches = []
    for field in CONFIG_CONTRACT_FIELDS:
        if actual.get(field) != expected.get(field):
            mismatches.append((field, expected.get(field), actual.get(field)))

    if mismatches:
        lines = [
            f"config preflight failed against {reference_path}",
            "Correct config.yaml to the published values; do not relax this comparison.",
        ]
        for field, exp, got in mismatches:
            lines.append(f"  {field}: expected {exp!r}, got {got!r}")
        fail("\n".join(lines))

    for field in CONFIG_REPORTED_DRIFT_FIELDS:
        if actual.get(field) != expected.get(field):
            print(
                "WARNING config differs from published reference: "
                f"{field}: expected {expected.get(field)!r}, got {actual.get(field)!r}. "
                "This run is not directly comparable on that field."
            )
    return expected


def checkpoint_role(cfg: Config) -> str:
    return (
        "diagnostic only; final configured epoch "
        f"{cfg.max_epochs} is fixed before evaluation and used unconditionally"
    )


def clear_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def severity_to_label(severity: pd.Series) -> pd.Series:
    return severity.astype(str).isin(POSITIVE_SEVERITIES).astype("int64")


def check_fixed_counts(y_train: np.ndarray, y_test: np.ndarray) -> None:
    train_pos = int(y_train.sum())
    test_pos = int(y_test.sum())
    if len(y_train) != EXPECTED_TRAIN_ROWS or train_pos != EXPECTED_TRAIN_POSITIVES:
        fail(
            f"train set expected {EXPECTED_TRAIN_ROWS} rows / {EXPECTED_TRAIN_POSITIVES} "
            f"positives, got {len(y_train)} rows / {train_pos} positives"
        )
    if len(y_test) != EXPECTED_TEST_ROWS or test_pos != EXPECTED_TEST_POSITIVES:
        fail(
            f"test set expected {EXPECTED_TEST_ROWS} rows / {EXPECTED_TEST_POSITIVES} "
            f"positives, got {len(y_test)} rows / {test_pos} positives"
        )


def load_fold_assignment(
    path: Path,
    train_ids: list[str],
    y_train: np.ndarray,
    test_ids: list[str],
    folds_arg: int,
) -> pd.DataFrame:
    if not path.is_file():
        fail(f"fold assignment not found: {path}")
    assignment = pd.read_csv(path, dtype={"node_id": str, "severity": str, "fold": int})
    expected_cols = ["node_id", "severity", "fold"]
    if list(assignment.columns) != expected_cols:
        fail(f"{path} columns must be exactly {expected_cols}, got {list(assignment.columns)}")
    if len(assignment) != len(train_ids):
        fail(f"fold assignment has {len(assignment)} rows, expected {len(train_ids)} train nodes")
    if assignment["node_id"].duplicated().any():
        dupes = assignment.loc[assignment["node_id"].duplicated(), "node_id"].head(3).tolist()
        fail(f"fold assignment has duplicate node_id values, e.g. {dupes}")

    train_set = set(train_ids)
    assign_set = set(assignment["node_id"])
    if assign_set != train_set:
        fail(
            "fold assignment does not cover train nodes exactly "
            f"({len(train_set - assign_set)} missing, {len(assign_set - train_set)} extra)"
        )

    overlap = train_set & set(test_ids)
    if overlap:
        fail(f"train/test sidecars overlap on {len(overlap)} node_ids, e.g. {sorted(overlap)[:3]}")

    unexpected = set(assignment["severity"].astype(str)) - VALID_SEVERITIES
    if unexpected:
        fail(f"fold assignment has unexpected severities: {sorted(unexpected)}")

    folds = sorted(int(f) for f in assignment["fold"].unique())
    expected_folds = list(range(len(folds)))
    if folds != expected_folds:
        fail(f"fold assignment must use contiguous folds {expected_folds}, got {folds}")
    if folds_arg != len(folds):
        fail(
            f"--folds is compatibility-only for this fixed partition; got {folds_arg}, "
            f"but {path.name} contains {len(folds)} distinct folds"
        )

    train_index = pd.DataFrame({
        "node_id": train_ids,
        "label": y_train.astype("int64"),
        "_row": np.arange(len(train_ids), dtype=np.int64),
    })
    merged = train_index.merge(assignment, on="node_id", how="left", validate="one_to_one")
    if merged["fold"].isna().any():
        fail("fold assignment merge lost train nodes")
    expected_label = severity_to_label(merged["severity"])
    bad_label = merged.loc[expected_label.to_numpy() != merged["label"].to_numpy()]
    if len(bad_label):
        example = bad_label[["node_id", "severity", "label"]].head(5).to_dict("records")
        fail(f"fold severity does not match row label, e.g. {example}")

    for fold in folds:
        fold_rows = merged[merged["fold"] == fold]
        if fold_rows.empty:
            fail(f"fold {fold} is empty")
        if int(fold_rows["label"].sum()) == 0:
            fail(f"fold {fold} has no positive validation node")

    return merged.sort_values("_row").reset_index(drop=True)


def load_inputs(cfg: Config, fold_assignment: Path, folds: int) -> Inputs:
    print("DATA")
    train_raw, y_train, train_ids = load_data("train", cfg)
    test_raw, y_test, test_ids = load_data("test", cfg)
    if train_ids is None:
        fail("data/train_node_ids.txt is required for fold assignment")
    if test_ids is None:
        fail("data/test_node_ids.txt is required for test score artifacts")
    if len(set(train_ids)) != len(train_ids):
        fail("data/train_node_ids.txt contains duplicate node IDs")
    if len(set(test_ids)) != len(test_ids):
        fail("data/test_node_ids.txt contains duplicate node IDs")
    check_fixed_counts(y_train, y_test)

    assignment_by_row = load_fold_assignment(fold_assignment, train_ids, y_train, test_ids, folds)
    print("  fold assignment:")
    print(
        assignment_by_row.groupby(["fold", "severity"]).size().unstack(fill_value=0).to_string()
    )

    return Inputs(
        train_texts=prepare_texts(train_raw, cfg),
        y_train=y_train.astype("int64"),
        train_ids=train_ids,
        test_texts=prepare_texts(test_raw, cfg),
        y_test=y_test.astype("int64"),
        test_ids=test_ids,
        assignment_by_row=assignment_by_row,
    )


def choose_threshold(scores: np.ndarray, target_recall: float) -> tuple[float, int]:
    if len(scores) == 0:
        fail("held-out fold has no positive scores to set the threshold")
    if not 0.0 < target_recall <= 1.0:
        fail("--target-recall must be in (0, 1]")
    sorted_scores = np.sort(scores.astype(float))
    wanted = int(np.ceil(len(sorted_scores) * target_recall))
    cutoff_index = len(sorted_scores) - wanted
    return float(sorted_scores[cutoff_index]), cutoff_index


def check_probability_scores(name: str, scores: np.ndarray) -> None:
    if len(scores) == 0:
        fail(f"{name}: no scores produced")
    if not np.isfinite(scores).all():
        fail(f"{name}: scores contain NaN or infinite values")
    lo, hi = float(scores.min()), float(scores.max())
    if lo < 0.0 or hi > 1.0:
        fail(
            f"{name}: score range [{lo:.6g}, {hi:.6g}] is outside [0, 1]; "
            "expected p_unsafe, not logits"
        )


def metric_block(labels: np.ndarray, scores: np.ndarray, flags: np.ndarray) -> dict:
    out = utils.compute_metrics(labels.astype("int64"), flags.astype("int64"))
    out.update(utils.compute_ranking_metrics(labels.astype("int64"), scores.astype(float)))
    return out


def flatten_metrics(prefix: str, metrics: dict) -> dict:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def write_training_artifacts(
    fold_dir: Path,
    cfg: Config,
    history: list[dict],
    train_count: int,
    train_positive_count: int,
) -> None:
    checkpoints = [h["checkpoint"] for h in history if h["checkpoint"]]
    final_checkpoint = fold_dir / "checkpoints" / f"head_epoch{cfg.max_epochs:02d}.pt"
    if not final_checkpoint.is_file():
        fail(f"final epoch checkpoint was not written: {final_checkpoint}")
    shutil.copy2(final_checkpoint, fold_dir / "final_head.pt")
    (fold_dir / "training.json").write_text(json.dumps(json_ready({
        "config": cfg.to_json(),
        "n_train": train_count,
        "n_train_unsafe": train_positive_count,
        "history": history,
        "checkpoints": checkpoints,
        "final_epoch_used": cfg.max_epochs,
        "final_checkpoint": final_checkpoint.name,
        "final_head": "final_head.pt",
        "checkpoint_role": checkpoint_role(cfg),
    }), indent=2), encoding="utf-8")


def load_eval_only_head(model: SafetyClassifier, cfg: Config, device: torch.device, path: Path) -> None:
    if not path.is_file():
        fail(f"eval-only checkpoint is missing: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("hidden") != cfg.hidden or ckpt.get("dim") != model.dim:
        fail(
            f"{path}: architecture mismatch (hidden={ckpt.get('hidden')}, dim={ckpt.get('dim')}) "
            f"vs config/model (hidden={cfg.hidden}, dim={model.dim})"
        )
    model.head.load_state_dict(ckpt["state_dict"])


def evaluate_fold(
    fold: int,
    fold_dir: Path,
    cfg: Config,
    model: SafetyClassifier,
    device: torch.device,
    inputs: Inputs,
    target_recall: float,
) -> dict:
    fold_values = inputs.assignment_by_row["fold"].to_numpy()
    val_idx = np.flatnonzero(fold_values == fold)
    train_idx = np.flatnonzero(fold_values != fold)
    validation_texts = [inputs.train_texts[i] for i in val_idx]
    validation_labels = inputs.y_train[val_idx]

    validation_scores = predict_proba(model, validation_texts, device, f"fold{fold}_validation")
    test_scores = predict_proba(model, inputs.test_texts, device, f"fold{fold}_test")
    throughput_texts = inputs.train_texts + inputs.test_texts
    throughput_t0 = time.perf_counter()
    _ = predict_proba(model, throughput_texts, device, f"fold{fold}_throughput")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    throughput_elapsed = time.perf_counter() - throughput_t0
    throughput_n_nodes = len(throughput_texts)
    throughput_nodes_per_sec = (
        throughput_n_nodes / throughput_elapsed if throughput_elapsed > 0 else 0.0
    )

    check_probability_scores(f"fold {fold} validation", validation_scores)
    check_probability_scores(f"fold {fold} test", test_scores)

    positive_scores = validation_scores[validation_labels == 1]
    threshold, cutoff_index = choose_threshold(positive_scores, target_recall)

    val_flags = validation_scores >= threshold
    val_metrics = utils.compute_metrics(validation_labels, val_flags.astype("int64"))
    ties = int((validation_scores == threshold).sum())
    pos_ties = int(((validation_scores == threshold) & (validation_labels == 1)).sum())
    tolerance = (1.0 + ties) / int(validation_labels.sum())
    if float(val_metrics["recall"]) - target_recall > tolerance:
        fail(
            f"fold {fold}: validation recall {val_metrics['recall']:.6f} is not within "
            f"one positive node plus threshold ties ({tolerance:.6f}) of "
            f"target {target_recall:.6f}; "
            f"ties_at_threshold={ties}, positive_ties_at_threshold={pos_ties}"
        )

    tuned_flags = test_scores >= threshold
    test_df = pd.DataFrame({
        "node_id": inputs.test_ids,
        "score": test_scores.astype(float),
        "flag_tuned": tuned_flags.astype(bool),
    })
    test_df.to_parquet(fold_dir / "test_scores.parquet", index=False)

    tuned = metric_block(inputs.y_test, test_scores, tuned_flags)

    threshold_info = {
        "fold": fold,
        "target_recall": target_recall,
        "threshold": threshold,
        "cutoff_index_in_sorted_positive_scores": cutoff_index,
        "validation_recall": val_metrics["recall"],
        "validation_positive_count": int(validation_labels.sum()),
        "validation_ties_at_threshold": int((validation_scores == threshold).sum()),
        "validation_positive_ties_at_threshold": int(
            ((validation_scores == threshold) & (validation_labels == 1)).sum()
        ),
        "test_ties_at_threshold": int((test_scores == threshold).sum()),
        "test_positive_ties_at_threshold": int(
            ((test_scores == threshold) & (inputs.y_test == 1)).sum()
        ),
        "flag_rule": "score >= threshold",
        "score_scale": "p_unsafe",
    }
    (fold_dir / "threshold.json").write_text(
        json.dumps(json_ready(threshold_info), indent=2), encoding="utf-8"
    )

    metrics = {
        "fold": fold,
        "training_rows": int(len(train_idx)),
        "training_positive_count": int(inputs.y_train[train_idx].sum()),
        "validation": val_metrics,
        "threshold": threshold_info,
        "tuned": tuned,
        "throughput_nodes_per_sec": throughput_nodes_per_sec,
        "throughput_n_nodes": int(throughput_n_nodes),
        "throughput_seconds": throughput_elapsed,
        "score_scale": "p_unsafe",
        "final_epoch_used": cfg.max_epochs,
        "checkpoint_role": checkpoint_role(cfg),
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


def load_summary_header(path: Path) -> list[str]:
    header = load_json(path)
    if not isinstance(header, list) or not all(isinstance(c, str) for c in header):
        fail(f"{path} must be a JSON list of column names")
    if len(header) != len(set(header)):
        fail(f"{path} contains duplicate column names")
    return header


def require_summary_header(df: pd.DataFrame, expected_header: list[str]) -> pd.DataFrame:
    actual = list(df.columns)
    missing = [c for c in expected_header if c not in actual]
    if missing:
        fail(f"summary columns do not include the contrast summary schema (missing={missing})")
    out = df[expected_header]
    if list(out.columns) != expected_header:
        fail("summary header ordering does not match the contrast summary schema")
    return out


def write_summary(model_dir: Path, rows: list[dict], header_path: Path) -> pd.DataFrame:
    load_summary_header(header_path)
    expected_header = CONTRAST_SUMMARY_HEADER
    df = require_summary_header(pd.DataFrame(rows), expected_header)
    numeric_cols = [c for c in df.columns if c != "fold" and pd.api.types.is_numeric_dtype(df[c])]
    mean = {"fold": "mean", **{c: df[c].mean() for c in numeric_cols}}
    std = {"fold": "std", **{c: df[c].std(ddof=1) for c in numeric_cols}}
    out = pd.concat([df, pd.DataFrame([mean, std])], ignore_index=True)
    out = require_summary_header(out, expected_header)
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
    result_names = {"summary.csv", "kfold_manifest.json", "fold_assignment.csv"} | {
        f"fold_{i}" for i in range(folds)
    }
    existing = [p for p in model_dir.iterdir() if p.name in result_names]
    if existing and not overwrite:
        fail(f"{model_dir} already contains k-fold outputs; pass --overwrite to replace them")
    for path in existing:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def package_versions() -> dict[str, str | None]:
    out = {"torch": torch.__version__}
    for dist in ("transformers", "sentence-transformers"):
        try:
            out[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            out[dist] = None
    return out


def gpu_info() -> dict:
    if not torch.cuda.is_available():
        return {"available": False}
    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": props.name,
        "total_memory_gb": props.total_memory / 2**30,
    }


def write_manifest(
    model_dir: Path,
    args: argparse.Namespace,
    cfg: Config,
    published_config: dict,
    fold_wall_seconds: list[dict],
    total_wall_seconds: float,
) -> None:
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "config": str(args.config.resolve()),
        "data_dir": str(cfg.data_dir),
        "fold_assignment": str(args.fold_assignment),
        "folds": args.folds,
        "target_recall": args.target_recall,
        "eval_only_from": str(args.eval_only_from.resolve()) if args.eval_only_from else None,
        "positive_severities": sorted(POSITIVE_SEVERITIES),
        "config_sha256": file_sha256(args.config),
        "fold_assignment_sha256": file_sha256(args.fold_assignment),
        "published_head_config_sha256": file_sha256(args.published_config),
        "contrast_summary_header_sha256": file_sha256(args.summary_header),
        "published_head_config": published_config,
        "config_actual": cfg.to_json(),
        "package_versions": package_versions(),
        "gpu": gpu_info(),
        "fold_wall_seconds": fold_wall_seconds,
        "total_wall_seconds": total_wall_seconds,
        "throughput": throughput_summary(model_dir, args.folds),
        "final_epoch_used": cfg.max_epochs,
        "checkpoint_role": checkpoint_role(cfg),
    }
    (model_dir / "kfold_manifest.json").write_text(
        json.dumps(json_ready(manifest), indent=2), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--model-config-name", default=None)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--fold-assignment", type=Path, default=None)
    parser.add_argument("--published-config", type=Path, default=None)
    parser.add_argument("--summary-header", type=Path, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--target-recall", type=float, default=0.8)
    parser.add_argument("--eval-only-from", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.fold_assignment is None:
        args.fold_assignment = args.data_dir / "fold_assignment.csv"
    if args.published_config is None:
        args.published_config = args.data_dir / "published_head_config.json"
    if args.summary_header is None:
        args.summary_header = args.data_dir / "contrast_summary_header.json"

    t_run = time.time()
    cfg = replace(Config.load(args.config), data_dir=args.data_dir.resolve())
    published_config = check_published_config(cfg, args.published_config)
    inputs = load_inputs(cfg, args.fold_assignment, args.folds)
    load_summary_header(args.summary_header)

    default_model_name = f"{cfg.variant}_h{cfg.hidden}"
    if args.eval_only_from is not None:
        default_model_name = f"{default_model_name}_eval_only"
    model_name = args.model_config_name or default_model_name
    model_dir = args.out_root / model_name
    guard_output_dir(model_dir, args.folds, args.overwrite)
    model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.fold_assignment, model_dir / "fold_assignment.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nMODEL  {cfg.model_name} on {device}  max_seq_length={cfg.max_seq_length}")

    rows: list[dict] = []
    fold_wall_seconds: list[dict] = []
    fold_values = inputs.assignment_by_row["fold"].to_numpy()
    for fold in range(args.folds):
        clear_cuda_cache()
        t_fold = time.time()
        print(f"\n=== fold_{fold} ===", flush=True)
        fold_dir = model_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = fold_dir / "checkpoints"

        train_idx = np.flatnonzero(fold_values != fold)
        train_texts = [inputs.train_texts[i] for i in train_idx]
        y_fold_train = inputs.y_train[train_idx]

        torch.manual_seed(cfg.seed)
        model = SafetyClassifier(cfg).to(device)
        if cfg.fp16 and device.type == "cuda":
            model.encoder.half()

        if args.eval_only_from is None:
            print(
                f"  training rows: {len(train_texts)} "
                f"({int(y_fold_train.sum())} unsafe / {int((1 - y_fold_train).sum())} safe)"
            )
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            history = train_head(model, device, train_texts, y_fold_train, cfg, ckpt_dir)
            write_training_artifacts(
                fold_dir,
                cfg,
                history,
                train_count=len(train_texts),
                train_positive_count=int(y_fold_train.sum()),
            )
        else:
            source_head = args.eval_only_from / f"fold_{fold}" / "final_head.pt"
            print(f"  eval-only: loading {source_head}; no training will run")
            load_eval_only_head(model, cfg, device, source_head)

        row = evaluate_fold(fold, fold_dir, cfg, model, device, inputs, args.target_recall)
        rows.append(row)
        print(
            f"  ap={row['tuned_average_precision']:.4f}, "
            f"roc={row['tuned_roc_auc']:.4f}, "
            f"precision={row['tuned_precision']:.4f}"
        )

        secs = time.time() - t_fold
        fold_wall_seconds.append({"fold": fold, "seconds": secs})
        print(f"  fold wall-clock: {secs / 60:.1f} min")
        del model
        clear_cuda_cache()

    summary = write_summary(model_dir, rows, args.summary_header)
    write_manifest(
        model_dir,
        args,
        cfg,
        published_config,
        fold_wall_seconds=fold_wall_seconds,
        total_wall_seconds=time.time() - t_run,
    )
    print_summary(summary)
    print(f"\nwrote {model_dir / 'summary.csv'}")
    print(f"wrote {model_dir / 'kfold_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
