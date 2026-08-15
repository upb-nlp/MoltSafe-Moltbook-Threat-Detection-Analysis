from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from . import manifest, paths
from .embedding import load_embedding_model
from .normalize import mask_emails
from .seeding import DEFAULT_SEED, set_seed
from .vectors import l2_normalize
import faiss
from tqdm import tqdm
from .scan import _chunk_by_tokens


logger = logging.getLogger(__name__)


def _maybe_mask(texts, mask: bool, placeholder: str):
    return [mask_emails(t, placeholder) for t in texts] if mask else list(texts)


def _chunk_cfg_from(cfg: Dict):
    ecfg = cfg.get("embedding", {})
    if not ecfg.get("chunking", False):
        return None
    return int(ecfg.get("chunk_tokens", 256)), int(ecfg.get("chunk_overlap", 32))


def _encode_texts(model, texts, batch_size: int, desc: str | None = None,
                  chunk_cfg=None) -> np.ndarray:

    if not chunk_cfg:
        rng = range(0, len(texts), batch_size)
        chunks = []
        for i in (tqdm(rng, desc=desc) if desc else rng):
            chunks.append(model.encode(texts[i: i + batch_size], show_progress_bar=False,
                                       convert_to_numpy=True))
        return l2_normalize(np.vstack(chunks).astype(np.float32))

    chunk_tokens, chunk_overlap = chunk_cfg
    tokenizer = model.tokenizer
    tokenizer.model_max_length = int(1e9)

    win_texts: list = []
    owners: list = []
    for j, t in enumerate(texts):
        for w in _chunk_by_tokens(t, tokenizer, chunk_tokens, chunk_overlap):
            win_texts.append(w)
            owners.append(j)
    owners_arr = np.asarray(owners)

    parts = []
    rng = range(0, len(win_texts), batch_size)
    for i in (tqdm(rng, desc=desc) if desc else rng):
        parts.append(model.encode(win_texts[i: i + batch_size], show_progress_bar=False,
                                   convert_to_numpy=True))
    wvecs = l2_normalize(np.vstack(parts).astype(np.float32))

    out = np.zeros((len(texts), wvecs.shape[1]), dtype=np.float32)
    for j in range(len(texts)):
        mine = np.where(owners_arr == j)[0]
        if len(mine):
            out[j] = wvecs[mine].mean(axis=0)
    return l2_normalize(out)


def _build_cluster_index(df: pd.DataFrame, embeddings: np.ndarray, dim: int,
                         n_clusters: int, seed: int, run_dir: Path,
                         index_path: Path, map_path: Path, faiss) -> int:
    n = len(embeddings)
    k = max(1, min(int(n_clusters), n))
    logger.info("Clustering %d attacks into %d centroids (spherical k-means) ...", n, k)
    kmeans = faiss.Kmeans(dim, k, niter=25, spherical=True, seed=int(seed), verbose=False)
    kmeans.train(embeddings)
    centroids = l2_normalize(kmeans.centroids.reshape(k, dim).astype(np.float32))

    index = faiss.IndexFlatIP(dim)
    index.add(centroids)
    faiss.write_index(index, str(index_path))
    map_path.write_text(json.dumps({str(i): f"cluster:{i}" for i in range(k)}), encoding="utf-8")

    sims, assign = index.search(embeddings, 1)
    assign, sims = assign.ravel(), sims.ravel()
    cats = df["category"].tolist() if "category" in df.columns else [""] * n
    texts = df["attack_text"].tolist()
    rows = []
    for ci in range(k):
        members = np.where(assign == ci)[0]
        size = int(len(members))
        if size:
            member_cats = [cats[m] for m in members]
            dominant = max(set(member_cats), key=member_cats.count)
            rep = int(members[np.argmax(sims[members])])
            rep_text = texts[rep]
        else:
            dominant, rep_text = "", ""
        rows.append({"cluster_id": f"cluster:{ci}", "category": dominant, "source": "cluster",
                     "rep_text": rep_text, "rep_label": f"{size} attacks", "size": size})
    pd.DataFrame(rows).to_parquet(paths.clusters_path(run_dir), index=False)
    logger.info("Wrote %d cluster labels -> clusters.parquet", k)
    return k


def _build_contrast_index(cfg: Dict, df: pd.DataFrame, model, batch_size: int,
                          mask: bool, placeholder: str, seed: int, run_dir: Path,
                          index_path: Path, map_path: Path, faiss) -> Tuple[int, int, int]:
    scfg = cfg.get("search", {})
    mal_sev = scfg.get("malicious_severities", ["invoked", "exfiltrated"])
    ben_sev = scfg.get("benign_severities")
    benign_n = int(scfg.get("benign_sample", 20000))
    n_all = len(df)
    df = df[df["split"] == "train"]
    if len(df) == 0:
        raise ValueError(
            f"Threat dictionary has no split=='train' rows (of {n_all}): "
            f"{paths.threat_dictionary_path(run_dir)}"
        )
    logger.info("Prototype pool: %d train rows of %d dictionary rows (%d held out)",
                len(df), n_all, n_all - len(df))

    if "severity" in df.columns:
        sub = df[df["severity"].isin(mal_sev)]
        if len(sub) == 0:
            logger.warning("No attacks match severities %s; falling back to ALL attacks.", mal_sev)
            sub = df
    else:
        sub = df
    chunk_cfg = _chunk_cfg_from(cfg)
    if chunk_cfg:
        logger.info("Chunking ON for prototypes: %d-token windows, %d overlap (full-text embed)",
                    chunk_cfg[0], chunk_cfg[1])
    mal_texts = _maybe_mask(sub["attack_text"].astype(str).tolist(), mask, placeholder)
    logger.info("Malicious prototype: %d attacks (severities=%s)", len(mal_texts), mal_sev)
    mal_emb = _encode_texts(model, mal_texts, batch_size, desc="encode-malicious", chunk_cfg=chunk_cfg)
    dim = int(mal_emb.shape[1])
    mal_centroid = l2_normalize(mal_emb.mean(axis=0, keepdims=True))


    if ben_sev and "severity" in df.columns and df["severity"].isin(ben_sev).any():
        ben_artifact = df[df["severity"].isin(ben_sev)]
        btexts = _maybe_mask(ben_artifact["attack_text"].fillna("").astype(str).tolist(), mask, placeholder)
        ben_source = "dictionary:" + "+".join(map(str, ben_sev))
        logger.info("Benign prototype: %d dictionary rows (severities=%s)", len(btexts), ben_sev)
    else:
        ben_artifact = pd.read_parquet(paths.moltbook_nodes_path(run_dir),
                                       columns=["node_id", "node_type", "embed_text"])
        ben_artifact = ben_artifact.sample(n=min(benign_n, len(ben_artifact)),
                                           random_state=seed).reset_index(drop=True)
        btexts = _maybe_mask(ben_artifact["embed_text"].fillna("").astype(str).tolist(), mask, placeholder)
        ben_source = "moltbook:sample"
        logger.info("Benign prototype: %d Moltbook nodes (seed=%d)", len(btexts), seed)
    ben_emb = _encode_texts(model, btexts, batch_size, desc="encode-benign", chunk_cfg=chunk_cfg)
    ben_centroid = l2_normalize(ben_emb.mean(axis=0, keepdims=True))

    ben_artifact.to_parquet(paths.benign_sample_path(run_dir), index=False)
    n_benign = len(btexts)

    cents = np.vstack([mal_centroid, ben_centroid]).astype(np.float32)
    index = faiss.IndexFlatIP(dim)
    index.add(cents)
    faiss.write_index(index, str(index_path))
    map_path.write_text(json.dumps({"0": "malicious", "1": "benign"}), encoding="utf-8")

    clusters_out = paths.clusters_path(run_dir)
    pd.DataFrame([
        {"cluster_id": "malicious", "category": "malicious", "source": "llmail:" + "+".join(mal_sev),
         "rep_text": f"<malicious prototype: mean of {len(mal_texts)} successful attacks>",
         "rep_label": f"{len(mal_texts)} attacks", "size": len(mal_texts)},
        {"cluster_id": "benign", "category": "benign", "source": ben_source,
         "rep_text": f"<benign prototype: mean of {n_benign} nodes>",
         "rep_label": f"{n_benign} samples", "size": n_benign},
    ]).to_parquet(clusters_out, index=False)
    logger.info("Contrast prototypes built: malicious(%d) vs benign(%d)", len(mal_texts), n_benign)
    return dim, len(mal_texts), n_benign


def build_index(cfg: Dict, run_dir: Path) -> Dict[str, Any]:

    set_seed(cfg.get("random_seed", DEFAULT_SEED))
    started = manifest.now_iso()
    paths.phase1_dir(run_dir, create=True)
    threat_path = paths.threat_dictionary_path(run_dir)
    index_path = paths.faiss_index_path(run_dir)
    map_path = paths.threat_id_map_path(run_dir)

    embed_cfg = cfg["embedding"]
    model_name = embed_cfg["model_name"]
    batch_size = embed_cfg.get("batch_size", 128)
    mask = embed_cfg.get("mask_emails", False)
    placeholder = embed_cfg.get("email_placeholder", "@")

    scfg = cfg.get("search", {})
    mode = scfg.get("mode", "example")
    n_clusters = scfg.get("n_clusters", 8)
    seed = cfg.get("random_seed", 42)

    logger.info("Loading threat dictionary ...")
    df = pd.read_parquet(threat_path)
    threat_ids = df["threat_id"].tolist()

    model = load_embedding_model(embed_cfg)

    extra: Dict[str, Any] = {}
    if mode == "contrast":
        dim, n_mal, n_ben = _build_contrast_index(
            cfg, df, model, batch_size, mask, placeholder, seed, run_dir, index_path, map_path, faiss)
        entry_count = 2
        extra = {"n_malicious": n_mal, "n_benign": n_ben,
                 "malicious_severities": scfg.get("malicious_severities", ["invoked", "exfiltrated"]),
                 "benign_sample": int(scfg.get("benign_sample", 20000)),
                 "n_train": int((df["split"] == "train").sum()),
                 "n_test": int((df["split"] == "test").sum())}
        split_manifest = threat_path.parent / "split_manifest.json"
        if split_manifest.exists():
            extra["split"] = json.loads(split_manifest.read_text(encoding="utf-8"))
    else:
        texts = _maybe_mask(df["attack_text"].astype(str).tolist(), mask, placeholder)
        if mask:
            logger.info("Masked email addresses in threat texts (placeholder=%r)", placeholder)
        embeddings = _encode_texts(model, texts, batch_size, desc="encode-threats",
                                   chunk_cfg=_chunk_cfg_from(cfg))
        dim = int(embeddings.shape[1])
        if mode == "cluster":
            n_cent = _build_cluster_index(df, embeddings, dim, n_clusters, seed,
                                          run_dir, index_path, map_path, faiss)
            entry_count = n_cent
            extra = {"n_clusters": n_cent}
        else:
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings)
            faiss.write_index(index, str(index_path))
            map_path.write_text(json.dumps({str(i): tid for i, tid in enumerate(threat_ids)}),
                                encoding="utf-8")
            entry_count = len(threat_ids)

    finished = manifest.now_iso()
    manifest.save(run_dir, {
        "build_index": {
            "started_at": started,
            "finished_at": finished,
            "model_name": model_name,
            "vector_dim": dim,
            "search_mode": mode,
            "index_entries": entry_count,
            "threat_count": len(threat_ids),
            "index_path": str(index_path),
            "mask_emails": mask,
            "email_placeholder": placeholder,
            **extra,
        },
        "software": manifest.software_versions(),
    })

    logger.info("Index ready [%s]: %d entries, dim=%d", mode, entry_count, dim)
    return {"threat_count": len(threat_ids), "dim": dim, "mode": mode, "entries": entry_count}
