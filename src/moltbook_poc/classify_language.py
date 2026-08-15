from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from . import manifest, paths
from .normalize import mask_emails
from .scan import _chunk_windows_by_tokens
from .seeding import DEFAULT_SEED, set_seed
from lingua import IsoCode639_1, Language, LanguageDetectorBuilder
from transformers import AutoTokenizer



logger = logging.getLogger(__name__)

UND = "und"


def _build_detector(lang_cfg: Dict):

    iso_codes = lang_cfg.get("languages")
    if iso_codes:
        langs = [Language.from_iso_code_639_1(getattr(IsoCode639_1, c.upper()))
                 for c in iso_codes]
        builder = LanguageDetectorBuilder.from_languages(*langs)
    else:
        builder = LanguageDetectorBuilder.from_all_languages()
    if lang_cfg.get("low_accuracy_mode", True):
        builder = builder.with_low_accuracy_mode()
    return builder.with_preloaded_language_models().build()


def _iso(lang) -> str:
    if lang is None:
        return UND
    return lang.iso_code_639_1.name.lower()


def _expand_node_chunks(texts: List[str], tokenizer, chunking: bool,
                        chunk_tokens: int, chunk_overlap: int):
    chunk_texts: List[str] = []
    owners: List[int] = []
    indices: List[int] = []
    nchars: List[int] = []
    starts: List[int] = []
    ends: List[int] = []
    for row_i, text in enumerate(tqdm(texts, desc="chunk", unit="node")):
        if chunking:
            windows = _chunk_windows_by_tokens(text, tokenizer, chunk_tokens, chunk_overlap)
        else:
            windows = [{"text": text, "start_token": 0, "end_token": 0}]
        for ci, w in enumerate(windows):
            chunk_texts.append(w["text"])
            owners.append(row_i)
            indices.append(ci)
            nchars.append(len(w["text"]))
            starts.append(int(w["start_token"]))
            ends.append(int(w["end_token"]))
    return chunk_texts, owners, indices, nchars, starts, ends


def _detect_chunks(detector, chunk_texts: List[str], nchars: List[int],
                   min_chars: int, batch_size: int) -> List[str]:
    labels: List[str] = []
    n = len(chunk_texts)
    for i in tqdm(range(0, n, batch_size), desc="detect", unit="batch"):
        sub = chunk_texts[i:i + batch_size]
        results = detector.detect_languages_in_parallel_of(sub)
        for nc, lang in zip(nchars[i:i + batch_size], results):
            labels.append(UND if nc < min_chars else _iso(lang))
    return labels


def english_only_mask(df: pd.DataFrame, lang_cfg: Dict) -> Optional[pd.Series]:
    if not lang_cfg.get("english_only", False):
        return None
    if "english_char_frac" not in df.columns:
        logger.warning("english_only set but 'english_char_frac' missing "
                      "(run language-filter first); scanning all nodes.")
        return None
    min_frac = float(lang_cfg.get("english_min_frac", 0.50))
    return df["english_char_frac"].fillna(0.0) >= min_frac


def _node_summary(owners, labels, nchars, n_nodes: int, mixed_min_frac: float):
    lang_chars: List[Dict[str, int]] = [defaultdict(int) for _ in range(n_nodes)]
    totals = [0] * n_nodes
    for owner, lang, nc in zip(owners, labels, nchars):
        lang_chars[owner][lang] += nc
        totals[owner] += nc

    summaries = []
    for i in range(n_nodes):
        dist = lang_chars[i]
        total = totals[i] or 1
        detected = {k: v for k, v in dist.items() if k != UND}
        if detected:
            dominant = max(detected, key=detected.get)
            dom_chars = detected[dominant]
        else:
            dominant, dom_chars = UND, 0
        en_chars = dist.get("en", 0)
        mixed = [k for k, v in detected.items() if v / total >= mixed_min_frac]
        summaries.append({
            "dominant_language": dominant,
            "dominant_char_frac": dom_chars / total,
            "english_char_frac": en_chars / total,
            "is_mixed_language": len(mixed) > 1,
            "n_chars": int(totals[i]),
            "lang_char_dist": json.dumps(dict(sorted(dist.items(),
                                                     key=lambda kv: kv[1], reverse=True))),
        })
    return summaries


def _distribution_csv(chunk_df: pd.DataFrame, node_df: pd.DataFrame, out_path: Path) -> None:
    rows: List[Dict[str, Any]] = []

    def _add(scope: str, frame: pd.DataFrame, weight_col: Optional[str], lang_col: str):
        for node_type, grp in [("all", frame), *frame.groupby("node_type")]:
            n_units = len(grp)
            chars = int(grp["n_chars"].sum()) if "n_chars" in grp else 0
            counts = grp.groupby(lang_col).size()
            char_by = grp.groupby(lang_col)["n_chars"].sum() if weight_col else None
            for lang, cnt in counts.sort_values(ascending=False).items():
                rows.append({
                    "scope": scope,
                    "node_type": node_type,
                    "language": lang,
                    "n_units": int(cnt),
                    "pct_units": round(100 * cnt / n_units, 4) if n_units else 0.0,
                    "n_chars": int(char_by[lang]) if char_by is not None else "",
                    "pct_chars": round(100 * char_by[lang] / chars, 4)
                                 if char_by is not None and chars else "",
                })

    _add("chunk", chunk_df, "n_chars", "language")
    _add("node_dominant", node_df, "n_chars", "dominant_language")
    pd.DataFrame(rows).to_csv(out_path, index=False)


def classify_language(cfg: Dict, run_dir: Path) -> Dict[str, Any]:
    import yaml

    set_seed(cfg.get("random_seed", DEFAULT_SEED))
    started = manifest.now_iso()
    lang_cfg = cfg.get("language", {})
    subdir = lang_cfg.get("output_subdir", paths.LANGUAGE_SUBDIR)
    out_dir = paths.language_dir(run_dir, create=True, subdir=subdir)

    nodes_path = paths.moltbook_nodes_path(run_dir)
    scan_path = paths.scan_results_path(run_dir)
    chunk_out = paths.chunk_languages_path(run_dir, subdir=subdir)
    node_out = paths.node_languages_path(run_dir, subdir=subdir)
    dist_out = paths.language_distribution_path(run_dir, subdir=subdir)

    (out_dir / "config_used.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    embed_cfg = cfg.get("embedding", {})
    mask = embed_cfg.get("mask_emails", False)
    placeholder = embed_cfg.get("email_placeholder", "@")
    chunking = embed_cfg.get("chunking", False)
    chunk_tokens = int(embed_cfg.get("chunk_tokens", 256))
    chunk_overlap = int(embed_cfg.get("chunk_overlap", 32))
    min_chars = int(lang_cfg.get("min_chars", 20))
    mixed_min_frac = float(lang_cfg.get("mixed_min_frac", 0.10))
    detect_batch = int(lang_cfg.get("detect_batch_size", 50000))

    logger.info("Loading nodes ...")
    nodes_df = pd.read_parquet(nodes_path, columns=["node_id", "node_type", "embed_text"])
    n_nodes = len(nodes_df)

    texts = nodes_df["embed_text"].fillna("").tolist()
    if mask:
        texts = [mask_emails(t, placeholder) for t in texts]

    tokenizer = None
    if chunking:
        tokenizer = AutoTokenizer.from_pretrained(embed_cfg["model_name"])
        tokenizer.model_max_length = int(1e9)
        logger.info("Chunking ON: %d-token windows, %d overlap (matches scan).",
                    chunk_tokens, chunk_overlap)

    logger.info("Expanding %d nodes into chunks ...", n_nodes)
    chunk_texts, owners, indices, nchars, starts, ends = _expand_node_chunks(
        texts, tokenizer, chunking, chunk_tokens, chunk_overlap)
    logger.info("%d chunks; detecting languages (low_accuracy=%s) ...",
                len(chunk_texts), lang_cfg.get("low_accuracy_mode", True))

    detector = _build_detector(lang_cfg)
    labels = _detect_chunks(detector, chunk_texts, nchars, min_chars, detect_batch)

    node_ids = nodes_df["node_id"].astype(str).tolist()
    node_types = nodes_df["node_type"].astype(str).tolist()

    chunk_df = pd.DataFrame({
        "node_id": [node_ids[o] for o in owners],
        "node_type": [node_types[o] for o in owners],
        "chunk_index": indices,
        "start_token": starts,
        "end_token": ends,
        "n_chars": nchars,
        "language": labels,
    })
    chunk_df.to_parquet(chunk_out, index=False)

    summaries = _node_summary(owners, labels, nchars, n_nodes, mixed_min_frac)
    node_df = pd.DataFrame(summaries)
    node_df.insert(0, "node_type", node_types)
    node_df.insert(0, "node_id", node_ids)
    node_df.to_parquet(node_out, index=False)

    _distribution_csv(chunk_df, node_df, dist_out)
    merged_scan = False
    if scan_path.exists():
        scan = pd.read_parquet(scan_path)
        scan["node_id"] = scan["node_id"].astype(str)
        keep = ["node_id", "dominant_language", "dominant_char_frac",
                "english_char_frac", "is_mixed_language"]
        scan = scan.drop(columns=[c for c in keep if c != "node_id" and c in scan.columns],
                         errors="ignore")
        scan = scan.merge(
            node_df[keep].rename(columns={"dominant_language": "language"}),
            on="node_id", how="left")
        if "trigger_chunk_index" in scan.columns:
            trig = chunk_df.set_index(["node_id", "chunk_index"])["language"]
            keys = list(zip(scan["node_id"], scan["trigger_chunk_index"].fillna(0).astype(int)))
            scan["trigger_language"] = pd.Series(
                [trig.get(k, UND) for k in keys], index=scan.index)
        scan.to_parquet(scan_path, index=False)
        merged_scan = True
        logger.info("Merged language columns into scan_results.parquet")
    else:
        logger.warning("scan_results.parquet not found; wrote language tables only.")

    dom_counts = node_df["dominant_language"].value_counts()
    non_en_nodes = int((node_df["dominant_language"] != "en").sum())
    mixed_nodes = int(node_df["is_mixed_language"].sum())
    eng_frac = lang_cfg.get("english_min_frac", 0.50)
    eng_dominant = int((node_df["english_char_frac"] >= float(eng_frac)).sum())

    english_corpus_out = None
    if lang_cfg.get("english_only", False):
        eng_ids = set(node_df.loc[
            node_df["english_char_frac"].fillna(0.0) >= float(eng_frac), "node_id"])
        full = pd.read_parquet(nodes_path)
        full["node_id"] = full["node_id"].astype(str)
        lang_cols = node_df[["node_id", "dominant_language", "dominant_char_frac",
                             "english_char_frac", "is_mixed_language"]]
        full_eng = full[full["node_id"].isin(eng_ids)].merge(lang_cols, on="node_id", how="left")
        english_corpus_out = out_dir / "moltbook_nodes_english.parquet"
        full_eng.to_parquet(english_corpus_out, index=False)
        logger.info("english_only: wrote %d/%d English-dominant nodes -> %s",
                    len(full_eng), len(full), english_corpus_out)

    finished = manifest.now_iso()
    manifest.save(run_dir, {
        "classify_language": {
            "started_at": started,
            "finished_at": finished,
            "nodes": n_nodes,
            "chunks": len(chunk_texts),
            "output_subdir": subdir,
            "low_accuracy_mode": lang_cfg.get("low_accuracy_mode", True),
            "min_chars": min_chars,
            "mixed_min_frac": mixed_min_frac,
            "english_min_frac": float(eng_frac),
            "english_dominant_nodes": eng_dominant,
            "non_english_dominant_nodes": non_en_nodes,
            "mixed_language_nodes": mixed_nodes,
            "top_dominant_languages": {str(k): int(v) for k, v in dom_counts.head(10).items()},
            "merged_into_scan": merged_scan,
            "output_chunk_table": str(chunk_out),
            "output_node_table": str(node_out),
            "output_distribution_csv": str(dist_out),
            "english_only": bool(lang_cfg.get("english_only", False)),
            "english_corpus": str(english_corpus_out) if english_corpus_out else None,
        }
    })

    logger.info("Language: %d nodes, %d English-dominant, %d non-English, %d mixed.",
                n_nodes, eng_dominant, non_en_nodes, mixed_nodes)

    return {
        "nodes": n_nodes,
        "chunks": len(chunk_texts),
        "english_dominant_nodes": eng_dominant,
        "non_english_dominant_nodes": non_en_nodes,
        "mixed_language_nodes": mixed_nodes,
        "node_table": str(node_out),
        "chunk_table": str(chunk_out),
        "distribution_csv": str(dist_out),
        "merged_into_scan": merged_scan,
        "english_corpus": str(english_corpus_out) if english_corpus_out else None,
    }
