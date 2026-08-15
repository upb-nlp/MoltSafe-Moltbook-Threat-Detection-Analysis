from __future__ import annotations

import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from . import manifest, paths
from .normalize import dedup_by_hash, text_sha256
from .seeding import DEFAULT_SEED, set_seed
import datasets as hf_datasets


logger = logging.getLogger(__name__)


def _resolve_hf_commit_hash(dataset_name: str, revision: str) -> str:
    try:
        from huggingface_hub import HfApi

        info = HfApi().repo_info(repo_id=dataset_name, repo_type="dataset", revision=revision)
        return str(info.sha or revision)
    except Exception:
        return revision


def _str_field(row: Dict, *candidates, default: str = "") -> str:
    for c in candidates:
        v = row.get(c)
        if v is not None and str(v).strip():
            return str(v).strip()
    return default


def _int_field(row: Dict, *candidates, default: int = 0) -> int:
    for c in candidates:
        v = row.get(c)
        if v is not None:
            try:
                return int(v)
            except (ValueError, TypeError):
                pass
    return default


def _unix_to_iso(value: str) -> str:
    if value.isdigit():
        return datetime.fromtimestamp(int(value), tz=timezone.utc).isoformat()
    return value


def _process_posts(dataset, max_chars: int) -> List[Dict]:
    records: List[Dict] = []
    for row in tqdm(dataset, desc="posts", unit="post"):
        row = dict(row)
        title = _str_field(row, "title")
        text = _str_field(row, "text", "body", "selftext", "content")
        if title and text:
            raw_embed = f"{title}\n\n{text}"
        elif title:
            raw_embed = title
        else:
            raw_embed = text
        if not raw_embed:
            continue

        sha = text_sha256(raw_embed)
        source_id = _str_field(row, "id", "post_id")
        created = _str_field(row, "created_utc", "created_at", "timestamp")
        if created:
            created = _unix_to_iso(created)

        score = _int_field(row, "score")
        records.append({
            "node_id": f"post:{source_id}",
            "conversation_id": source_id,
            "node_type": "post",
            "source_id": source_id,
            "parent_source_id": None,
            "author_id": _str_field(row, "author", "author_id", "user_id") or None,
            "submolt_name": _str_field(row, "subreddit", "submolt_name", "submolt", "community") or None,
            "created_at": created or None,
            "depth": 0,
            "score": score,
            "upvotes": _int_field(row, "ups", "upvotes", default=score),
            "downvotes": _int_field(row, "downs", "downvotes"),
            "reply_count": _int_field(row, "num_comments", "reply_count", "comment_count"),
            "title": title or None,
            "text": text,
            "embed_text": raw_embed[:max_chars] if max_chars and max_chars > 0 else raw_embed,
            "text_sha256": sha,
        })
    return records


def _process_comments(dataset, max_chars: int) -> List[Dict]:
    records: List[Dict] = []
    for row in tqdm(dataset, desc="comments", unit="comment"):
        row = dict(row)
        text = _str_field(row, "text", "body", "content", "comment")
        if not text:
            continue

        sha = text_sha256(text)
        source_id = _str_field(row, "id", "comment_id")
        post_id = _str_field(row, "post_id", "link_id", "parent_id", "submission_id")
        if post_id.startswith("t3_"):
            post_id = post_id[3:]
        created = _str_field(row, "created_utc", "created_at", "timestamp")
        if created:
            created = _unix_to_iso(created)

        score = _int_field(row, "score")
        records.append({
            "node_id": f"comment:{source_id}",
            "conversation_id": post_id or source_id,
            "node_type": "comment",
            "source_id": source_id,
            "parent_source_id": post_id or None,
            "author_id": _str_field(row, "author", "author_id", "user_id") or None,
            "submolt_name": None,
            "created_at": created or None,
            "depth": _int_field(row, "depth", "level"),
            "score": score,
            "upvotes": _int_field(row, "ups", "upvotes", default=score),
            "downvotes": _int_field(row, "downs", "downvotes"),
            "reply_count": _int_field(row, "num_replies", "reply_count"),
            "title": None,
            "text": text,
            "embed_text": text[:max_chars] if max_chars and max_chars > 0 else text,
            "text_sha256": sha,
        })
    return records


def prepare_moltbook(cfg: Dict, run_dir: Path) -> Dict[str, Any]:

    set_seed(cfg.get("random_seed", DEFAULT_SEED))
    started = manifest.now_iso()
    out_path = paths.moltbook_nodes_path(run_dir)
    paths.phase1_dir(run_dir, create=True)

    mb = cfg["moltbook"]
    hf_name = mb["hf_dataset"]
    hf_rev = mb["hf_revision"]
    split = mb["split"]
    max_items = mb.get("max_items", 0)
    max_chars = mb.get("text_max_chars", 0)
    seed = cfg.get("random_seed", 42)
    if int(max_chars) != 0:
        raise ValueError("moltbook.text_max_chars must be 0 for reproducible text_sha256 verification")

    logger.info("Downloading posts from %s ...", hf_name)
    posts_ds = hf_datasets.load_dataset(
        hf_name, mb["post_config"],
        revision=hf_rev, split=split,
    )
    logger.info("Downloading comments from %s ...", hf_name)
    comments_ds = hf_datasets.load_dataset(
        hf_name, mb["comment_config"],
        revision=hf_rev, split=split,
    )

    commit_hash = _resolve_hf_commit_hash(hf_name, hf_rev)

    logger.info("Processing posts ...")
    posts = _process_posts(posts_ds, max_chars)
    raw_post_count = len(posts)
    posts = dedup_by_hash(posts, "text_sha256", "score")
    logger.info("Posts: %d raw -> %d after dedup", raw_post_count, len(posts))

    logger.info("Processing comments ...")
    comments = _process_comments(comments_ds, max_chars)
    raw_comment_count = len(comments)
    comments = dedup_by_hash(comments, "text_sha256", "score")
    logger.info("Comments: %d raw -> %d after dedup", raw_comment_count, len(comments))

    all_records = posts + comments
    if max_items and len(all_records) > max_items:
        rng = random.Random(seed)
        all_records = rng.sample(all_records, max_items)
        logger.info("Sampled %d of %d records", max_items, len(all_records) + max_items)

    df = pd.DataFrame(all_records)
    for col in ("depth", "score", "upvotes", "downvotes", "reply_count"):
        if col in df.columns:
            df[col] = pd.array(df[col], dtype="Int64")

    df.to_parquet(out_path, index=False)
    fingerprint = manifest.file_sha256(out_path)
    finished = manifest.now_iso()

    manifest.save(run_dir, {
        "prepare_moltbook": {
            "started_at": started,
            "finished_at": finished,
            "hf_dataset": hf_name,
            "hf_revision": hf_rev,
            "hf_commit_hash": commit_hash,
            "post_count": len(posts),
            "comment_count": len(comments),
            "total_rows": len(df),
            "output_fingerprint": fingerprint,
        }
    })

    logger.info("Wrote %d rows to %s", len(df), out_path)
    return {"rows": len(df), "output": str(out_path)}
