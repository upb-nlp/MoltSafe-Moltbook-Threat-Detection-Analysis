from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from moltbook_poc import repo_paths

REPO = repo_paths.root()
DEFAULT_CORPUS = repo_paths.path("corpus_english")
DEFAULT_OUTPUT = repo_paths.path("synced_nodes")
DEFAULT_JUDGE_RESULTS_OUTPUT = repo_paths.path("judge_results")
DEFAULT_HF_REPO = "PaulCl/MoltSafe-10K"
DEFAULT_HF_FILENAME = "dataset.csv"
DEFAULT_HF_REVISION = "main"

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
CORPUS_COLUMNS = SCAN_NODE_COLUMNS + ["title", "text"]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def text_value(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def build_embed_text(row: pd.Series) -> str:
    node_type = str(row["node_type"])
    title = text_value(row.get("title"))
    text = text_value(row.get("text"))
    if node_type == "post":
        if title and text:
            return f"{title}\n\n{text}"
        return title or text
    if node_type == "comment":
        return text
    raise ValueError(f"unexpected node_type for {row['node_id']}: {node_type}")


def read_id_list(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        ids = pd.read_csv(path, dtype=str, keep_default_na=False)
        if "node_id" not in ids.columns:
            raise ValueError(f"{path} must include a node_id column")
        keep = ["node_id"]
        if "text_sha256" in ids.columns:
            keep.append("text_sha256")
        ids = ids[keep].copy()
    else:
        rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        ids = pd.DataFrame({"node_id": [row for row in rows if row]})
    ids["node_id"] = ids["node_id"].astype(str)
    if ids["node_id"].duplicated().any():
        examples = ids.loc[ids["node_id"].duplicated(), "node_id"].head(5).tolist()
        raise ValueError(f"{path} contains duplicate node_id values, e.g. {examples}")
    return ids


def download_hf_dataset(repo_id: str, filename: str, revision: str,
                        cache_dir: Path | None = None) -> tuple[Path, dict[str, Any]]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download the published label dataset; "
            "install the project environment or pass --node-ids for a local CSV"
        ) from exc

    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "filename": filename,
        "revision": revision,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir

    print(f"downloading Hugging Face dataset {repo_id}/{filename} revision={revision}")
    path = Path(hf_hub_download(**kwargs))
    snapshot_revision = path.parent.name if path.parent.parent.name == "snapshots" else revision
    return path, {
        "source_type": "huggingface_dataset",
        "repo_id": repo_id,
        "filename": filename,
        "requested_revision": revision,
        "snapshot_revision": snapshot_revision,
        "cache_path": str(path),
    }


def resolve_published_dataset(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    if args.node_ids is not None:
        return args.node_ids, {
            "source_type": "local_file",
            "path": str(args.node_ids),
        }
    return download_hf_dataset(
        args.hf_repo,
        args.hf_filename,
        args.hf_revision,
        cache_dir=args.hf_cache_dir,
    )


def write_normalized_judge_results(source: Path, output: Path) -> dict | None:
    if source.suffix.lower() != ".csv":
        return None

    df = pd.read_csv(source)
    if "severity" not in df.columns:
        return None

    output.parent.mkdir(parents=True, exist_ok=True)
    severity = pd.to_numeric(df["severity"], errors="coerce").astype("Int64").astype("string")
    df["severity"] = severity.fillna("").astype(str)
    df.to_csv(output, index=False)
    return {
        "output": str(output),
        "output_sha256": file_sha256(output),
        "n_rows": int(len(df)),
    }


def sync_published_nodes(node_ids: Path, corpus: Path, output: Path,
                         dataset_source: dict[str, Any] | None = None) -> dict:
    ids = read_id_list(node_ids)
    nodes = pd.read_parquet(corpus, columns=CORPUS_COLUMNS)
    nodes["node_id"] = nodes["node_id"].astype(str)

    out = ids.merge(nodes, on="node_id", how="left", validate="one_to_one")
    missing = out.loc[out["node_type"].isna(), "node_id"].tolist()
    if missing:
        shown = ", ".join(missing[:20])
        raise ValueError(f"{len(missing)} requested node_id values were missing from corpus: {shown}")
    out["embed_text"] = out.apply(build_embed_text, axis=1)

    n_hash_mismatch = 0
    if "text_sha256" in out.columns:
        expected = out["text_sha256"].astype(str)
        actual = out["embed_text"].map(text_sha256)
        bad = expected.ne(actual)
        n_hash_mismatch = int(bad.sum())
        if n_hash_mismatch:
            shown = ", ".join(out.loc[bad, "node_id"].head(20).tolist())
            raise ValueError(f"{n_hash_mismatch} text_sha256 mismatches: {shown}")

    output.parent.mkdir(parents=True, exist_ok=True)
    out[SCAN_NODE_COLUMNS].to_parquet(output, index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus": str(corpus),
        "corpus_sha256": file_sha256(corpus),
        "id_list": str(node_ids),
        "id_list_sha256": file_sha256(node_ids),
        "published_dataset_source": dataset_source or {
            "source_type": "local_file",
            "path": str(node_ids),
        },
        "n_requested": int(len(ids)),
        "n_resolved": int(len(out)),
        "n_hash_mismatch": n_hash_mismatch,
        "output": str(output),
        "output_sha256": file_sha256(output),
    }
    (output.parent / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-ids", type=Path, default=None,
                        help=f"Local published dataset CSV/id list. If omitted, downloads {DEFAULT_HF_REPO}.")
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO,
                        help=f"Hugging Face dataset repo to download when --node-ids is omitted.")
    parser.add_argument("--hf-filename", default=DEFAULT_HF_FILENAME)
    parser.add_argument("--hf-revision", default=DEFAULT_HF_REVISION)
    parser.add_argument("--hf-cache-dir", type=Path, default=None)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--judge-results-output", type=Path, default=DEFAULT_JUDGE_RESULTS_OUTPUT)
    args = parser.parse_args()

    try:
        node_ids, dataset_source = resolve_published_dataset(args)
        manifest = sync_published_nodes(node_ids, args.corpus, args.output, dataset_source)
        judge_results = write_normalized_judge_results(node_ids, args.judge_results_output)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"synced {manifest['n_resolved']}/{manifest['n_requested']} nodes, "
        f"hash_mismatch={manifest['n_hash_mismatch']} -> {args.output}"
    )
    print(f"published dataset source -> {manifest['published_dataset_source']['source_type']}")
    if judge_results is not None:
        print(f"normalized judge results -> {args.judge_results_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
