from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from . import manifest, paths
from .embedding import load_embedding_model
from .normalize import mask_emails
from .seeding import DEFAULT_SEED, set_seed
from .vectors import l2_normalize

import faiss
from tqdm import tqdm

logger = logging.getLogger(__name__)

_TEXT_PREVIEW_CHARS = 200


def _clean_str(value: Any) -> str:
    return "" if pd.isna(value) else str(value)


def _clean_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return 0 if pd.isna(value) else int(value)
    except (TypeError, ValueError):
        return 0


def _threshold_flags(max_sim: float, thresholds: List[float]) -> str:
    return json.dumps({str(t): bool(max_sim >= t) for t in thresholds})


def _chunk_windows_by_tokens(text: str, tokenizer, chunk_tokens: int, overlap: int) -> List[Dict[str, Any]]:

    if not text:
        return [{"text": "", "start_token": 0, "end_token": 0}]
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= chunk_tokens:
        return [{"text": text, "start_token": 0, "end_token": len(ids)}]
    step = max(1, chunk_tokens - overlap)
    chunks: List[Dict[str, Any]] = []
    for start in range(0, len(ids), step):
        end = min(start + chunk_tokens, len(ids))
        chunks.append({
            "text": tokenizer.decode(ids[start:end], skip_special_tokens=True),
            "start_token": start,
            "end_token": end,
        })
        if end >= len(ids):
            break
    return chunks


def _chunk_by_tokens(text: str, tokenizer, chunk_tokens: int, overlap: int) -> List[str]:
    return [chunk["text"] for chunk in _chunk_windows_by_tokens(text, tokenizer, chunk_tokens, overlap)]


def _node_fields(node) -> Dict:
    return {
        "node_id": str(node.get("node_id", "")),
        "node_type": str(node.get("node_type", "")),
        "upvotes": _clean_int(node.get("upvotes")),
        "downvotes": _clean_int(node.get("downvotes")),
        "score": _clean_int(node.get("score")),
        "depth": _clean_int(node.get("depth")),
        "reply_count": _clean_int(node.get("reply_count")),
        "submolt_name": _clean_str(node.get("submolt_name")),
        "created_at": _clean_str(node.get("created_at")),
    }


def _trigger_fields(trigger_chunk_text: str = "", trigger_chunk_index: int = 0,
                    trigger_start_token: int | None = None,
                    trigger_end_token: int | None = None) -> Dict[str, Any]:
    return {
        "trigger_chunk_index": int(trigger_chunk_index),
        "trigger_start_token": trigger_start_token,
        "trigger_end_token": trigger_end_token,
        "trigger_chunk_text": trigger_chunk_text,
    }


def _result_row(node, row_scores, row_idx, id_map: Dict[str, str],
                threat_lookup: Dict[str, tuple], thresholds: List[float],
                num_chunks: int, trigger_chunk_text: str = "",
                trigger_chunk_index: int = 0, trigger_start_token: int | None = None,
                trigger_end_token: int | None = None) -> Dict:
    max_sim = float(row_scores[0]) if len(row_scores) > 0 else 0.0
    top_threat_ids = [id_map[str(k)] for k in row_idx if k >= 0]
    top_scores = [float(s) for s in row_scores if s > -1]
    nearest_tid = top_threat_ids[0] if top_threat_ids else ""
    nearest_cat, nearest_src = threat_lookup.get(nearest_tid, ("", ""))
    return {
        **_node_fields(node),
        "max_similarity": max_sim,
        "nearest_threat_id": nearest_tid,
        "nearest_category": nearest_cat,
        "nearest_source": nearest_src,
        "topk_threat_ids": json.dumps(top_threat_ids),
        "topk_scores": json.dumps(top_scores),
        "threshold_flags": _threshold_flags(max_sim, thresholds),
        "num_chunks": int(num_chunks),
        "text_preview": str(node.get("embed_text", ""))[:_TEXT_PREVIEW_CHARS],
        **_trigger_fields(trigger_chunk_text, trigger_chunk_index,
                          trigger_start_token, trigger_end_token),
    }


def _contrast_row(node, contrast: float, sim_mal: float, sim_ben: float,
                  threat_lookup: Dict[str, tuple], thresholds: List[float],
                  num_chunks: int, trigger_chunk_text: str = "",
                  trigger_chunk_index: int = 0, trigger_start_token: int | None = None,
                  trigger_end_token: int | None = None) -> Dict:

    nearest = "malicious" if contrast >= 0 else "benign"
    cat, src = threat_lookup.get(nearest, (nearest, "contrast"))
    return {
        **_node_fields(node),
        "max_similarity": float(contrast),
        "sim_malicious": float(sim_mal),
        "sim_benign": float(sim_ben),
        "nearest_threat_id": nearest,
        "nearest_category": cat,
        "nearest_source": src,
        "topk_threat_ids": json.dumps(["malicious", "benign"]),
        "topk_scores": json.dumps([float(sim_mal), float(sim_ben)]),
        "threshold_flags": _threshold_flags(float(contrast), thresholds),
        "num_chunks": int(num_chunks),
        "text_preview": str(node.get("embed_text", ""))[:_TEXT_PREVIEW_CHARS],
        **_trigger_fields(trigger_chunk_text, trigger_chunk_index,
                          trigger_start_token, trigger_end_token),
    }


def _expand_chunks(batch_texts, tokenizer, chunk_tokens: int, overlap: int):
    chunk_texts, owners, chunk_indices, starts, ends = [], [], [], [], []
    for j, t in enumerate(batch_texts):
        for chunk_index, chunk in enumerate(_chunk_windows_by_tokens(t, tokenizer, chunk_tokens, overlap)):
            chunk_texts.append(chunk["text"])
            owners.append(j)
            chunk_indices.append(chunk_index)
            starts.append(chunk["start_token"])
            ends.append(chunk["end_token"])
    return (chunk_texts, np.asarray(owners), np.asarray(chunk_indices),
            np.asarray(starts), np.asarray(ends))


def scan_moltbook(cfg: Dict, run_dir: Path, nodes_path: Path | str | None = None) -> Dict[str, Any]:


    set_seed(cfg.get("random_seed", DEFAULT_SEED))
    started = manifest.now_iso()
    paths.phase1_dir(run_dir, create=True)
    nodes_path = Path(nodes_path) if nodes_path is not None else paths.moltbook_nodes_path(run_dir)
    index_path = paths.faiss_index_path(run_dir)
    map_path = paths.threat_id_map_path(run_dir)
    threat_path = paths.threat_dictionary_path(run_dir)
    out_path = paths.scan_results_path(run_dir)

    embed_cfg = cfg["embedding"]
    model_name = embed_cfg["model_name"]
    batch_size = embed_cfg.get("batch_size", 128)
    mask = embed_cfg.get("mask_emails", False)
    placeholder = embed_cfg.get("email_placeholder", "@")
    chunking = embed_cfg.get("chunking", False)
    chunk_tokens = int(embed_cfg.get("chunk_tokens", 256))
    chunk_overlap = int(embed_cfg.get("chunk_overlap", 32))
    mode = cfg.get("search", {}).get("mode", "example")
    top_k = cfg["faiss"]["top_k"]
    thresholds = cfg.get("report", {}).get("thresholds", [])

    logger.info("Loading nodes, index, and labels (mode=%s) ...", mode)
    nodes_df = pd.read_parquet(nodes_path)
    required_node_cols = {"node_id", "embed_text"}
    missing_node_cols = sorted(required_node_cols - set(nodes_df.columns))
    if missing_node_cols:
        raise ValueError(
            f"{nodes_path}: missing required scan column(s): {', '.join(missing_node_cols)}"
        )
    index = faiss.read_index(str(index_path))
    id_map: Dict[str, str] = json.loads(map_path.read_text(encoding="utf-8"))

    if mode in ("cluster", "contrast"):
        label_df = pd.read_parquet(paths.clusters_path(run_dir))
        threat_lookup: Dict[str, tuple] = {
            row["cluster_id"]: (row["category"], row["source"])
            for _, row in label_df.iterrows()
        }
    else:
        threat_df = pd.read_parquet(threat_path)
        threat_lookup = {
            row["threat_id"]: (row["category"], row["source"])
            for _, row in threat_df.iterrows()
        }

    model = load_embedding_model(embed_cfg)
    texts = nodes_df["embed_text"].fillna("").tolist()
    if mask:
        texts = [mask_emails(t, placeholder) for t in texts]
        logger.info("Masked email addresses in node texts (placeholder=%r)", placeholder)
    n = len(texts)
    tokenizer = model.tokenizer if chunking else None
    if chunking:
        tokenizer.model_max_length = int(1e9)
        logger.info("Chunking ON: %d-token windows, %d overlap; node score = max over its chunks",
                    chunk_tokens, chunk_overlap)
    contrast_cents = None
    if mode == "contrast":
        contrast_cents = np.vstack([index.reconstruct(0), index.reconstruct(1)]).astype(np.float32)
        logger.info("Contrast mode: score = cos(node, malicious) - cos(node, benign)")
    logger.info("Scanning %d nodes ...", n)

    def _encode_norm(text_list):
        v = model.encode(text_list, batch_size=batch_size, show_progress_bar=False,
                         convert_to_numpy=True)
        return l2_normalize(v.astype(np.float32))

    results: List[Dict] = []
    t0 = time.perf_counter()

    for i in tqdm(range(0, n, batch_size), desc="scan"):
        batch_texts = texts[i: i + batch_size]
        batch_rows = nodes_df.iloc[i: i + batch_size]

        if chunking:
            chunk_texts, owners, chunk_indices, starts, ends = _expand_chunks(
                batch_texts, tokenizer, chunk_tokens, chunk_overlap)
            cvecs = _encode_norm(chunk_texts)
            if mode == "contrast":
                sims = cvecs @ contrast_cents.T
                contrast = sims[:, 0] - sims[:, 1]
                for j in range(len(batch_texts)):
                    mine = np.where(owners == j)[0]
                    best = mine[int(np.argmax(contrast[mine]))]
                    results.append(_contrast_row(
                        batch_rows.iloc[j], contrast[best], sims[best, 0], sims[best, 1],
                        threat_lookup, thresholds, num_chunks=len(mine),
                        trigger_chunk_text=chunk_texts[best],
                        trigger_chunk_index=int(chunk_indices[best]),
                        trigger_start_token=int(starts[best]),
                        trigger_end_token=int(ends[best])))
            else:
                cscores, cidx = index.search(cvecs, top_k)
                for j in range(len(batch_texts)):
                    mine = np.where(owners == j)[0]
                    best = mine[int(np.argmax(cscores[mine, 0]))]
                    results.append(_result_row(
                        batch_rows.iloc[j], cscores[best], cidx[best],
                        id_map, threat_lookup, thresholds, num_chunks=len(mine),
                        trigger_chunk_text=chunk_texts[best],
                        trigger_chunk_index=int(chunk_indices[best]),
                        trigger_start_token=int(starts[best]),
                        trigger_end_token=int(ends[best])))
        else:
            vecs = _encode_norm(batch_texts)
            if mode == "contrast":
                sims = vecs @ contrast_cents.T
                contrast = sims[:, 0] - sims[:, 1]
                for j in range(len(batch_texts)):
                    results.append(_contrast_row(
                        batch_rows.iloc[j], contrast[j], sims[j, 0], sims[j, 1],
                        threat_lookup, thresholds, num_chunks=1,
                        trigger_chunk_text=batch_texts[j],
                        trigger_chunk_index=0))
            else:
                scores_mat, idx_mat = index.search(vecs, top_k)
                for j, (row_scores, row_idx) in enumerate(zip(scores_mat, idx_mat)):
                    results.append(_result_row(
                        batch_rows.iloc[j], row_scores, row_idx,
                        id_map, threat_lookup, thresholds, num_chunks=1,
                        trigger_chunk_text=batch_texts[j],
                        trigger_chunk_index=0))

    elapsed = time.perf_counter() - t0
    rows_per_sec = n / elapsed if elapsed > 0 else 0.0

    out_df = pd.DataFrame(results)
    out_df.to_parquet(out_path, index=False)
    fingerprint = manifest.file_sha256(out_path)
    finished = manifest.now_iso()

    logger.info("Scan complete: %.1f rows/sec", rows_per_sec)
    manifest.save(run_dir, {
        "scan_moltbook": {
            "started_at": started,
            "finished_at": finished,
            "rows_scanned": n,
            "rows_per_sec": round(rows_per_sec, 2),
            "top_k": top_k,
            "model_name": model_name,
            "max_seq_length": embed_cfg.get("max_seq_length"),
            "chunking": chunking,
            "chunk_tokens": chunk_tokens if chunking else None,
            "chunk_overlap": chunk_overlap if chunking else None,
            "nodes_path": str(nodes_path),
            "output_fingerprint": fingerprint,
        }
    })

    return {"rows": n, "rows_per_sec": round(rows_per_sec, 2), "output": str(out_path)}
