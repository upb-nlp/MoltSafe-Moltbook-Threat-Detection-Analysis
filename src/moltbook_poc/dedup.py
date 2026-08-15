from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from . import manifest, paths
from .seeding import DEFAULT_SEED, set_seed
import pyarrow.parquet as pq
from .classify_language import english_only_mask

logger = logging.getLogger(__name__)


def dedup_by_post(cfg: Dict, run_dir: Path) -> Dict[str, Any]:
    set_seed(cfg.get("random_seed", DEFAULT_SEED))
    started = manifest.now_iso()

    scan_path = paths.scan_results_path(run_dir)
    nodes_path = paths.moltbook_nodes_path(run_dir)
    out_path = paths.flagged_by_post_path(run_dir)

    contrast_min = float(cfg.get("dedup", {}).get("contrast_min", 0.0))

    logger.info("Loading scan results and node map ...")

    cols = ["node_id", "node_type", "submolt_name", "max_similarity"]
    if "english_char_frac" in set(pq.ParquetFile(scan_path).schema_arrow.names):
        cols.append("english_char_frac")
    scan = pd.read_parquet(scan_path, columns=cols)
    nodes = pd.read_parquet(nodes_path, columns=["node_id", "conversation_id"])
    node_to_conv = dict(zip(nodes["node_id"].astype(str),
                            nodes["conversation_id"].astype(str)))

    eng_mask = english_only_mask(scan, cfg.get("language", {}))
    if eng_mask is not None:
        kept = int(eng_mask.sum())
        logger.info("english_only: keeping %d/%d English-dominant nodes", kept, len(scan))
        scan = scan[eng_mask]

    flagged = scan[scan["max_similarity"] >= contrast_min].copy()
    flagged["node_id"] = flagged["node_id"].astype(str)
    flagged["conversation_id"] = flagged["node_id"].map(node_to_conv)

    missing = int(flagged["conversation_id"].isna().sum())
    if missing:
        logger.warning("%d flagged node(s) had no conversation_id in nodes; dropped.", missing)
        flagged = flagged.dropna(subset=["conversation_id"])

    flagged = flagged.sort_values(["conversation_id", "node_id"], kind="stable")

    cols = ["conversation_id", "post_node_id", "submolt_name", "n_flagged",
            "flagged_node_ids", "flagged_node_types", "flagged_scores"]
    rows = []
    for conv_id, grp in flagged.groupby("conversation_id", sort=True):
        submolts = [s for s in grp["submolt_name"].tolist() if isinstance(s, str) and s]
        rows.append({
            "conversation_id": conv_id,
            "post_node_id": f"post:{conv_id}",
            "submolt_name": submolts[0] if submolts else "",
            "n_flagged": int(len(grp)),
            "flagged_node_ids": grp["node_id"].tolist(),
            "flagged_node_types": grp["node_type"].astype(str).tolist(),
            "flagged_scores": [float(x) for x in grp["max_similarity"].tolist()],
        })

    out_df = pd.DataFrame(rows, columns=cols)
    if not out_df.empty:
        out_df = out_df.sort_values(
            ["n_flagged", "conversation_id"], ascending=[False, True]
        ).reset_index(drop=True)

    paths.phase1_dir(run_dir, create=True)
    out_df.to_parquet(out_path, index=False)
    fingerprint = manifest.file_sha256(out_path)
    finished = manifest.now_iso()

    n_flagged = int(len(flagged))
    n_posts = int(len(out_df))
    multi = int((out_df["n_flagged"] > 1).sum()) if n_posts else 0
    max_flags = int(out_df["n_flagged"].max()) if n_posts else 0

    logger.info("Dedup: %d flagged node(s) -> %d post(s) (%d multi-flag, max %d/post)",
                n_flagged, n_posts, multi, max_flags)
    manifest.save(run_dir, {
        "dedup": {
            "started_at": started,
            "finished_at": finished,
            "contrast_min": contrast_min,
            "flagged_nodes": n_flagged,
            "unique_posts": n_posts,
            "multi_flag_posts": multi,
            "max_flags_per_post": max_flags,
            "output_fingerprint": fingerprint,
        }
    })

    return {"flagged_nodes": n_flagged, "unique_posts": n_posts, "output": str(out_path)}
