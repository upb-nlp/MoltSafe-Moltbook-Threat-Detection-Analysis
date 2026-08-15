from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from . import manifest, paths
from .normalize import mask_emails
from .seeding import DEFAULT_SEED, set_seed
from .classify_language import english_only_mask


logger = logging.getLogger(__name__)

_STRATA = [
    ("high_confidence", 0.55, 1.01),
    ("mid_confidence", 0.50, 0.55),
    ("borderline", 0.45, 0.50),
    ("near_miss", 0.40, 0.45),
    ("negative_control", 0.00, 0.40),
]

_REVIEW_COLS = [
    "node_id", "node_type", "text_preview", "max_similarity",
    "trigger_chunk_index", "trigger_start_token", "trigger_end_token", "trigger_chunk_text",
    "nearest_category", "nearest_source", "nearest_threat_id",
    "upvotes", "downvotes", "score", "submolt_name",
]

_TOP_TEXT_CHARS = 220
_TOP_ATTACK_CHARS = 160


def _build_top_matches(df: pd.DataFrame, run_dir: Path, top_n: int,
                       mode: str = "example") -> pd.DataFrame:

    top = df.nlargest(top_n, "max_similarity").copy()
    top.insert(0, "rank", range(1, len(top) + 1))

    try:
        nodes = pd.read_parquet(paths.moltbook_nodes_path(run_dir),
                                columns=["node_id", "title", "text"])
        top = top.merge(nodes, on="node_id", how="left")
    except Exception as exc:  
        logger.warning("Top matches: could not join node text (%s)", exc)
        top["title"], top["text"] = None, top.get("text_preview")

    if mode in ("cluster", "contrast"):
        src_path, key, text_col, label_col = paths.clusters_path(run_dir), "cluster_id", "rep_text", "rep_label"
    else:
        src_path, key, text_col, label_col = paths.threat_dictionary_path(run_dir), "threat_id", "attack_text", "scenario"
    try:
        th = pd.read_parquet(src_path)
        keep = [c for c in (key, text_col, label_col) if c in th.columns]
        th = th[keep].rename(columns={key: "nearest_threat_id",
                                      text_col: "nearest_attack_text",
                                      label_col: "nearest_scenario"})
        top = top.merge(th, on="nearest_threat_id", how="left")
    except Exception as exc:  
        logger.warning("Top matches: could not join match text (%s)", exc)
        top["nearest_attack_text"], top["nearest_scenario"] = None, None

    return top


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip().replace("\n", " ")


def _print_top_matches(top: pd.DataFrame) -> None:
    print(f"\n=== Top {len(top)} most attack-similar examples ===")
    for _, r in top.iterrows():
        sim = float(r["max_similarity"])
        sub = _as_text(r.get("submolt_name"))
        title = _as_text(r.get("title"))
        text = _as_text(r.get("text")) or _as_text(r.get("text_preview"))
        trigger = _as_text(r.get("trigger_chunk_text"))
        scen = _as_text(r.get("nearest_scenario"))
        atk = _as_text(r.get("nearest_attack_text"))

        head = f"#{int(r['rank']):>2}  sim={sim:.3f}  {str(r['node_type']):<7} nearest={r['nearest_category']}"
        if sub and sub.lower() not in ("nan", "none"):
            head += f"  (m/{sub})"
        print(head)
        if title:
            print(f"     TITLE: {title[:_TOP_TEXT_CHARS]}")
        if trigger:
            print(f"     CHUNK: {trigger[:_TOP_TEXT_CHARS]}")
        print(f"     TEXT : {text[:_TOP_TEXT_CHARS]}")
        print(f"     ~ATTACK[{scen}]: {atk[:_TOP_ATTACK_CHARS]}")


def generate_report(cfg: Dict, run_dir: Path) -> Dict[str, Any]:
    set_seed(cfg.get("random_seed", DEFAULT_SEED))
    started = manifest.now_iso()
    scan_path = paths.scan_results_path(run_dir)
    summary_path = run_dir / "threshold_summary.csv"
    sample_path = run_dir / "review_sample.csv"
    top_path = run_dir / "top_matches.csv"

    thresholds: List[float] = cfg["report"]["thresholds"]
    n_sample = cfg["report"].get("review_sample_per_stratum", 50)
    top_n = cfg["report"].get("top_examples", 25)
    seed = cfg.get("random_seed", 42)
    rng = random.Random(seed)

    logger.info("Loading scan results ...")
    df = pd.read_parquet(scan_path)

    eng_mask = english_only_mask(df, cfg.get("language", {}))
    if eng_mask is not None:
        logger.info("english_only: keeping %d/%d English-dominant nodes",
                    int(eng_mask.sum()), len(df))
        df = df[eng_mask].reset_index(drop=True)

    total = len(df)

    summary_rows: List[Dict] = []
    for t in thresholds:
        flagged = df[df["max_similarity"] >= t]
        by_type = flagged["node_type"].value_counts().to_dict()
        by_cat = flagged["nearest_category"].value_counts().to_dict()
        summary_rows.append({
            "threshold": t,
            "flagged_count": len(flagged),
            "flagged_pct": round(100 * len(flagged) / total, 4) if total else 0.0,
            "post_count": by_type.get("post", 0),
            "comment_count": by_type.get("comment", 0),
            "categories_json": json.dumps(by_cat),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)
    logger.info("Threshold summary written to %s", summary_path)

    sample_rows: List[pd.DataFrame] = []
    for stratum_name, lo, hi in _STRATA:
        mask = (df["max_similarity"] >= lo) & (df["max_similarity"] < hi)
        pool = df[mask]
        k = min(n_sample, len(pool))
        if k > 0:
            chosen = pool.sample(n=k, random_state=seed)
        else:
            chosen = pool
        chosen = chosen.copy()
        chosen["stratum"] = stratum_name
        sample_rows.append(chosen)
        logger.info("Stratum %s: pool=%d, sampled=%d", stratum_name, len(pool), k)

    review_cols = _REVIEW_COLS + ["stratum"]
    if sample_rows:
        review_df = pd.concat(sample_rows, ignore_index=True)
        available = [c for c in review_cols if c in review_df.columns]
        review_df[available].to_csv(sample_path, index=False)
    else:
        pd.DataFrame(columns=review_cols).to_csv(sample_path, index=False)

    logger.info("Review sample written to %s", sample_path)

    mode = cfg.get("search", {}).get("mode", "example")
    top = _build_top_matches(df, run_dir, top_n, mode)

    embed_cfg = cfg.get("embedding", {})
    if embed_cfg.get("mask_emails", False) and "nearest_attack_text" in top.columns:
        ph = embed_cfg.get("email_placeholder", "@")
        top["nearest_attack_text"] = top["nearest_attack_text"].apply(
            lambda t: mask_emails(t, ph) if isinstance(t, str) else t)
    top_csv_cols = [
        "rank", "max_similarity", "node_id", "node_type", "nearest_category",
        "nearest_source", "submolt_name",
        "trigger_chunk_index", "trigger_start_token", "trigger_end_token", "trigger_chunk_text",
        "title", "text",
        "nearest_threat_id", "nearest_scenario", "nearest_attack_text",
    ]
    top[[c for c in top_csv_cols if c in top.columns]].to_csv(top_path, index=False)
    logger.info("Top %d matches written to %s", len(top), top_path)

    print("\n=== Threshold Summary ===")
    print(f"{'Threshold':>10}  {'Flagged':>8}  {'Pct':>7}")
    for row in summary_rows:
        print(f"{row['threshold']:>10.2f}  {row['flagged_count']:>8}  {row['flagged_pct']:>6.2f}%")

    _print_top_matches(top)

    finished = manifest.now_iso()
    manifest.save(run_dir, {
        "report": {
            "started_at": started,
            "finished_at": finished,
            "total_nodes": total,
            "thresholds": thresholds,
            "n_sample_per_stratum": n_sample,
        }
    })

    return {
        "total_nodes": total,
        "threshold_summary": str(summary_path),
        "review_sample": str(sample_path),
        "top_matches": str(top_path),
    }
