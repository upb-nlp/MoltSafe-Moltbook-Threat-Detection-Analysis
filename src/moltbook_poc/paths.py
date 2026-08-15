from __future__ import annotations
from pathlib import Path

PHASE1_SUBDIR = "phase1"


def phase1_dir(run_dir: Path, create: bool = False) -> Path:
    d = Path(run_dir) / PHASE1_SUBDIR
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def moltbook_nodes_path(run_dir: Path) -> Path:
    return phase1_dir(run_dir) / "moltbook_nodes.parquet"


def threat_dictionary_path(run_dir: Path) -> Path:
    return phase1_dir(run_dir) / "threat_dictionary.parquet"


def faiss_index_path(run_dir: Path) -> Path:
    return phase1_dir(run_dir) / "faiss_attack.index"


def threat_id_map_path(run_dir: Path) -> Path:
    return phase1_dir(run_dir) / "threat_id_map.json"


def clusters_path(run_dir: Path) -> Path:
    return phase1_dir(run_dir) / "clusters.parquet"


def benign_sample_path(run_dir: Path) -> Path:
    return phase1_dir(run_dir) / "benign_sample.parquet"


def scan_results_path(run_dir: Path) -> Path:
    return phase1_dir(run_dir) / "scan_results.parquet"


def flagged_by_post_path(run_dir: Path) -> Path:
    return phase1_dir(run_dir) / "flagged_by_post.parquet"


LANGUAGE_SUBDIR = "language_analysis"


def language_dir(run_dir: Path, create: bool = False,
                 subdir: str = LANGUAGE_SUBDIR) -> Path:
    d = Path(run_dir) / subdir
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def chunk_languages_path(run_dir: Path, subdir: str = LANGUAGE_SUBDIR) -> Path:
    return language_dir(run_dir, subdir=subdir) / "chunk_languages.parquet"


def node_languages_path(run_dir: Path, subdir: str = LANGUAGE_SUBDIR) -> Path:
    return language_dir(run_dir, subdir=subdir) / "node_languages.parquet"


def language_distribution_path(run_dir: Path, subdir: str = LANGUAGE_SUBDIR) -> Path:
    return language_dir(run_dir, subdir=subdir) / "language_distribution.csv"
