from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from moltbook_poc import repo_paths

REPO = repo_paths.root()
DEFAULT_SEEDS = repo_paths.path("fold_data_dir")
DEFAULT_OUT_DIR = repo_paths.path("fold_data_dir")

UNSAFE_SEVERITIES = {"3", "4", "5"}
EXPECTED_COUNTS = {"train": (639, 5438), "test": (273, 2330)}
SPLITS = ("train", "test")

PUBLISHED_HEAD_CONFIG = {
    "model_name": "Qwen/Qwen3-Embedding-0.6B",
    "max_seq_length": 8192,
    "chunking": False,
    "fp16": True,
    "batch_size": 16,
    "hidden": 0,
    "dropout": 0.2,
    "lr": 0.001,
    "weight_decay": 0.01,
    "max_epochs": 15,
    "checkpoint_every": 2,
    "num_classes": 2,
    "mask_emails": True,
    "seed": 20260725,
    "variant": "noprefix",
    "instruct_prefix": None,
}

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


def build_split(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"node_id", "severity", "attack_text"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")

    severity = df["severity"].astype(str)
    unknown = sorted(set(severity) - UNSAFE_SEVERITIES - {"0"})
    if unknown:
        raise ValueError(f"{path}: unexpected severity values {unknown}")

    return pd.DataFrame({
        "node_id": df["node_id"].astype(str),
        "text": df["attack_text"].astype(str),
        "label": severity.isin(UNSAFE_SEVERITIES).astype("int64"),
    })


def check_counts(split: str, df: pd.DataFrame) -> None:
    n_unsafe = int((df["label"] == 1).sum())
    n_safe = int((df["label"] == 0).sum())
    expected = EXPECTED_COUNTS[split]
    if (n_unsafe, n_safe) != expected:
        raise ValueError(f"{split}: label counts {(n_unsafe, n_safe)} != expected {expected}")
    if df["text"].isna().any():
        raise ValueError(f"{split}: null text")
    if df["text"].str.strip().eq("").any():
        raise ValueError(f"{split}: empty text")


def write_outputs(seeds: Path, out_dir: Path) -> None:
    frames = {split: build_split(seeds / f"{split}_seeds.parquet") for split in SPLITS}
    overlap = set(frames["train"]["node_id"]) & set(frames["test"]["node_id"])
    if overlap:
        raise ValueError(f"train/test share {len(overlap)} node_ids")

    out_dir.mkdir(parents=True, exist_ok=True)
    for split, frame in frames.items():
        check_counts(split, frame)
        frame[["text", "label"]].to_parquet(out_dir / f"{split}.parquet", index=False)
        (out_dir / f"{split}_node_ids.txt").write_text(
            "\n".join(frame["node_id"].tolist()), encoding="utf-8"
        )
        n_unsafe = int(frame["label"].sum())
        n_safe = int((1 - frame["label"]).sum())
        print(f"{split}: {len(frame)} rows, unsafe={n_unsafe}, safe={n_safe}")


def copy_sidecar(source: Path, out_dir: Path, name: str) -> None:
    target = out_dir / name
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    print(f"{name}: {target}")


def write_json_sidecar(out_dir: Path, name: str, payload) -> None:
    target = out_dir / name
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"{name}: {target}")


def check_sidecars(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required sidecar(s) not found: {missing}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=Path, default=DEFAULT_SEEDS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fold-assignment", type=Path, default=None)
    parser.add_argument("--published-config", type=Path, default=None)
    parser.add_argument("--summary-header", type=Path, default=None)
    parser.add_argument(
        "--sidecar-profile",
        choices=["copy", "kaggle-training"],
        default="kaggle-training",
        help="copy sidecar JSONs from paths, or generate the fixed Kaggle training contracts",
    )
    args = parser.parse_args()
    if args.fold_assignment is None:
        args.fold_assignment = args.out_dir / "fold_assignment.csv"
    if args.published_config is None:
        args.published_config = args.out_dir / "published_head_config.json"
    if args.summary_header is None:
        args.summary_header = args.out_dir / "contrast_summary_header.json"

    sidecars = [args.fold_assignment]
    if args.sidecar_profile == "copy":
        sidecars.extend([args.published_config, args.summary_header])
    check_sidecars(sidecars)

    write_outputs(args.seeds, args.out_dir)
    copy_sidecar(args.fold_assignment, args.out_dir, "fold_assignment.csv")
    if args.sidecar_profile == "kaggle-training":
        write_json_sidecar(args.out_dir, "published_head_config.json", PUBLISHED_HEAD_CONFIG)
        write_json_sidecar(args.out_dir, "contrast_summary_header.json", CONTRAST_SUMMARY_HEADER)
    else:
        copy_sidecar(args.published_config, args.out_dir, "published_head_config.json")
        copy_sidecar(args.summary_header, args.out_dir, "contrast_summary_header.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
