from __future__ import annotations

import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import click
import yaml

from .prepare_moltbook import prepare_moltbook
from .build_index import build_index
from .scan import scan_moltbook
from .classify_language import classify_language
from . import repo_paths


for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

for _noisy in ("faiss", "faiss.loader", "httpx", "huggingface_hub"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

_REPO = repo_paths.root()
_DEFAULT_CONFIG = repo_paths.config("corpus.yaml")
_DEFAULT_DATA_DIR = repo_paths.path("data_dir")
_DEFAULT_FOLD_DATA_DIR = repo_paths.path("fold_data_dir")
_DEFAULT_CORPUS_RUN_ID = "corpus"
_DEFAULT_BATCH_PROMPT = _REPO / "data_prep" / "judge_api" / "prompt.txt"


def _load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_run_dir(run_id: str | None, base: Path, overwrite: bool) -> Path:
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = base / run_id
    if run_dir.exists() and not overwrite:
        click.echo(f"ERROR: Run folder already exists: {run_dir}\n"
                   f"Pass --overwrite to reuse it.", err=True)
        sys.exit(1)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _run_script(relative_path: str, args: list[str]) -> None:
    command = [sys.executable, str(_REPO / relative_path), *args]
    result = subprocess.run(command, cwd=_REPO, check=False)
    if result.returncode:
        raise click.exceptions.Exit(result.returncode)


def _path_arg(path: Path) -> str:
    return str(Path(path))


@click.group()
def cli() -> None:
    """"""


@cli.command("prep-data")
@click.option("--config", "config_path", default=str(_DEFAULT_CONFIG), show_default=True)
@click.option("--runs-dir", type=click.Path(file_okay=False, path_type=Path),
              default=_DEFAULT_DATA_DIR, show_default=True)
@click.option("--run-id", default=_DEFAULT_CORPUS_RUN_ID, show_default=True)
@click.option("--overwrite", is_flag=True)
def cmd_prep_data(config_path, runs_dir, run_id, overwrite):
    cfg = _load_config(config_path)
    run_dir = _resolve_run_dir(run_id, Path(runs_dir), overwrite)
    result = prepare_moltbook(cfg, run_dir)
    click.echo(f"Done. {result['rows']} rows -> {result['output']}")


@cli.command("language-filter")
@click.option("--config", "config_path", default=str(_DEFAULT_CONFIG), show_default=True)
@click.option("--runs-dir", type=click.Path(file_okay=False, path_type=Path),
              default=_DEFAULT_DATA_DIR, show_default=True)
@click.option("--run-id", default=_DEFAULT_CORPUS_RUN_ID, show_default=True)
@click.option("--overwrite", is_flag=True)
def cmd_language_filter(config_path, runs_dir, run_id, overwrite):
    cfg = _load_config(config_path)
    run_dir = _resolve_run_dir(run_id, Path(runs_dir), overwrite)
    result = classify_language(cfg, run_dir)
    click.echo(f"Done. {result['nodes']} nodes / {result['chunks']} chunks -> "
               f"{result['node_table']}")
    click.echo(f"English-dominant={result['english_dominant_nodes']}  "
               f"non-English={result['non_english_dominant_nodes']}  "
               f"mixed={result['mixed_language_nodes']}")
    click.echo(f"Distribution -> {result['distribution_csv']}")
    if result.get("english_corpus"):
        click.echo(f"English-only corpus (sample from this) -> {result['english_corpus']}")


@cli.command("judge-sample")
@click.option("--n", type=int, required=True, help="How many nodes to sample.")
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--parquet", type=click.Path(dir_okay=False, path_type=Path),
              default=repo_paths.path("corpus_english"), show_default=True)
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path),
              default=repo_paths.path("judge_sample_input"), show_default=True)
def cmd_judge_sample(n, seed, parquet, output):
    _run_script("data_prep/judge_api/sample_input.py", [
        "--n", str(n),
        "--seed", str(seed),
        "--parquet", _path_arg(parquet),
        "--output", _path_arg(output),
    ])


@cli.command("judge-batch-submit")
@click.option("--input", "input_csv", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--prompt", type=click.Path(dir_okay=False, path_type=Path),
              default=_DEFAULT_BATCH_PROMPT, show_default=True)
@click.option("--model", default=None, help="Override the judge model used by the batch script.")
@click.option("--effort", default=None, help="Override the reasoning effort used by the batch script.")
@click.option("--dry-run", is_flag=True)
@click.option("--force", is_flag=True)
def cmd_judge_batch_submit(input_csv, output_dir, prompt, model, effort, dry_run, force):
    args = [
        "--input", _path_arg(input_csv),
        "--output-dir", _path_arg(output_dir),
        "--prompt", _path_arg(prompt),
    ]
    if model is not None:
        args.extend(["--model", model])
    if effort is not None:
        args.extend(["--effort", effort])
    if dry_run:
        args.append("--dry-run")
    if force:
        args.append("--force")
    _run_script("data_prep/judge_api/batch_api/submit_batch.py", args)


@cli.command("judge-batch-status")
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
def cmd_judge_batch_status(output_dir):
    _run_script("data_prep/judge_api/batch_api/check_status.py", [
        "--output-dir", _path_arg(output_dir),
    ])


@cli.command("judge-batch-fetch")
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
def cmd_judge_batch_fetch(output_dir):
    _run_script("data_prep/judge_api/batch_api/fetch_results.py", [
        "--output-dir", _path_arg(output_dir),
    ])


@cli.command("sync-published-nodes")
@click.option("--node-ids", type=click.Path(dir_okay=False, path_type=Path),
              default=None, help="Local published dataset CSV/id list; overrides Hugging Face download.")
@click.option("--hf-repo", default="PaulCl/MoltSafe-10K", show_default=True)
@click.option("--hf-filename", default="dataset.csv", show_default=True)
@click.option("--hf-revision", default="main", show_default=True)
@click.option("--hf-cache-dir", type=click.Path(file_okay=False, path_type=Path), default=None)
@click.option("--corpus", type=click.Path(dir_okay=False, path_type=Path),
              default=repo_paths.path("corpus_english"), show_default=True)
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path),
              default=repo_paths.path("synced_nodes"), show_default=True)
@click.option("--judge-results-output", type=click.Path(dir_okay=False, path_type=Path),
              default=repo_paths.path("judge_results"), show_default=True)
def cmd_sync_published_nodes(node_ids, hf_repo, hf_filename, hf_revision, hf_cache_dir,
                             corpus, output, judge_results_output):
    args = [
        "--hf-repo", hf_repo,
        "--hf-filename", hf_filename,
        "--hf-revision", hf_revision,
        "--corpus", _path_arg(corpus),
        "--output", _path_arg(output),
        "--judge-results-output", _path_arg(judge_results_output),
    ]
    if node_ids is not None:
        args.extend(["--node-ids", _path_arg(node_ids)])
    if hf_cache_dir is not None:
        args.extend(["--hf-cache-dir", _path_arg(hf_cache_dir)])
    _run_script("data_prep/sync_published_nodes.py", args)


@cli.command("prepare-contrast-data")
@click.option("--results", type=click.Path(dir_okay=False, path_type=Path),
              default=repo_paths.path("judge_results"), show_default=True)
@click.option("--corpus", type=click.Path(dir_okay=False, path_type=Path),
              default=repo_paths.path("synced_nodes"), show_default=True)
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path),
              default=_DEFAULT_FOLD_DATA_DIR, show_default=True)
@click.option("--severity", "severities", type=int, multiple=True,
              default=(0, 3, 4, 5), show_default=True)
@click.option("--test-fraction", type=float, default=0.30, show_default=True)
@click.option("--split-seed", type=int, default=20260720, show_default=True)
@click.option("--fold-seed", type=int, default=42, show_default=True)
@click.option("--folds", type=int, default=5, show_default=True)
@click.option("--overwrite", is_flag=True)
def cmd_prepare_contrast_data(results, corpus, out_dir, severities,
                              test_fraction, split_seed, fold_seed, folds, overwrite):
    args = [
        "--results", _path_arg(results),
        "--corpus", _path_arg(corpus),
        "--out-dir", _path_arg(out_dir),
        "--severity", *[str(severity) for severity in severities],
        "--test-fraction", str(test_fraction),
        "--split-seed", str(split_seed),
        "--fold-seed", str(fold_seed),
        "--folds", str(folds),
    ]
    if overwrite:
        args.append("--overwrite")
    _run_script("data_prep/build_judge10k_dict.py", args)


@cli.command("eval-contrast")
@click.option("--config", "config_path", type=click.Path(dir_okay=False, path_type=Path),
              required=True)
@click.option("--out-root", type=click.Path(file_okay=False, path_type=Path),
              default=repo_paths.path("results_dir"), show_default=True)
@click.option("--model-config-name", default=None)
@click.option("--data-dir", type=click.Path(file_okay=False, path_type=Path),
              default=_DEFAULT_FOLD_DATA_DIR, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--folds", type=int, default=5, show_default=True)
@click.option("--target-recall", type=float, default=0.8, show_default=True)
@click.option("--overwrite", is_flag=True)
def cmd_eval_contrast(config_path, out_root, model_config_name, data_dir,
                      seed, folds, target_recall, overwrite):
    args = [
        "--config", _path_arg(config_path),
        "--out-root", _path_arg(out_root),
        "--data-dir", _path_arg(data_dir),
        "--seed", str(seed),
        "--folds", str(folds),
        "--target-recall", str(target_recall),
    ]
    if model_config_name is not None:
        args.extend(["--model-config-name", model_config_name])
    if overwrite:
        args.append("--overwrite")
    _run_script("experiments/run_kfold_contrast.py", args)


@cli.command("prepare-head-eval-data")
@click.option("--seeds", type=click.Path(file_okay=False, path_type=Path),
              default=_DEFAULT_FOLD_DATA_DIR, show_default=True)
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path),
              default=_DEFAULT_FOLD_DATA_DIR, show_default=True)
@click.option("--fold-assignment", type=click.Path(dir_okay=False, path_type=Path),
              default=None, show_default="<out-dir>\\fold_assignment.csv")
@click.option("--published-config", type=click.Path(dir_okay=False, path_type=Path),
              default=None, show_default="<out-dir>\\published_head_config.json")
@click.option("--summary-header", type=click.Path(dir_okay=False, path_type=Path),
              default=None, show_default="<out-dir>\\contrast_summary_header.json")
@click.option("--sidecar-profile", type=click.Choice(["copy", "kaggle-training"]),
              default="kaggle-training", show_default=True)
def cmd_prepare_head_eval_data(seeds, out_dir, fold_assignment, published_config, summary_header,
                               sidecar_profile):
    args = [
        "--seeds", _path_arg(seeds),
        "--out-dir", _path_arg(out_dir),
        "--sidecar-profile", sidecar_profile,
    ]
    if fold_assignment is not None:
        args.extend(["--fold-assignment", _path_arg(fold_assignment)])
    if published_config is not None:
        args.extend(["--published-config", _path_arg(published_config)])
    if summary_header is not None:
        args.extend(["--summary-header", _path_arg(summary_header)])
    _run_script("data_prep/prepare_head_data.py", args)


@cli.command("eval-head")
@click.option("--config", "config_path", type=click.Path(dir_okay=False, path_type=Path),
              default=_REPO / "trained_classifier" / "config.yaml", show_default=True)
@click.option("--out-root", type=click.Path(file_okay=False, path_type=Path),
              default=_REPO / "trained_classifier" / "runs_kfold", show_default=True)
@click.option("--model-config-name", default=None)
@click.option("--data-dir", type=click.Path(file_okay=False, path_type=Path),
              default=_DEFAULT_FOLD_DATA_DIR, show_default=True)
@click.option("--fold-assignment", type=click.Path(dir_okay=False, path_type=Path),
              default=None, show_default="<data-dir>\\fold_assignment.csv")
@click.option("--published-config", type=click.Path(dir_okay=False, path_type=Path),
              default=None, show_default="<data-dir>\\published_head_config.json")
@click.option("--summary-header", type=click.Path(dir_okay=False, path_type=Path),
              default=None, show_default="<data-dir>\\contrast_summary_header.json")
@click.option("--folds", type=int, default=5, show_default=True)
@click.option("--target-recall", type=float, default=0.8, show_default=True)
@click.option("--eval-only-from", type=click.Path(file_okay=False, path_type=Path),
              default=_REPO / "trained_classifier" / "runs_kfold" / "noprefix_h0",
              show_default=True)
@click.option("--overwrite", is_flag=True)
def cmd_eval_head(config_path, out_root, model_config_name, data_dir, fold_assignment,
                  published_config, summary_header, folds, target_recall,
                  eval_only_from, overwrite):
    args = [
        "--config", _path_arg(config_path),
        "--out-root", _path_arg(out_root),
        "--data-dir", _path_arg(data_dir),
        "--folds", str(folds),
        "--target-recall", str(target_recall),
        "--eval-only-from", _path_arg(eval_only_from),
    ]
    if fold_assignment is not None:
        args.extend(["--fold-assignment", _path_arg(fold_assignment)])
    if published_config is not None:
        args.extend(["--published-config", _path_arg(published_config)])
    if summary_header is not None:
        args.extend(["--summary-header", _path_arg(summary_header)])
    if model_config_name is not None:
        args.extend(["--model-config-name", model_config_name])
    if overwrite:
        args.append("--overwrite")
    _run_script("trained_classifier/run_kfold_head.py", args)


@cli.command("build-index", hidden=True)
@click.option("--run-id", required=True)
@click.option("--config", "config_path", default=str(_DEFAULT_CONFIG), show_default=True)
@click.option("--runs-dir", type=click.Path(file_okay=False, path_type=Path),
              default=_DEFAULT_DATA_DIR, show_default=True)
@click.option("--overwrite", is_flag=True)
def cmd_build_index(run_id, config_path, runs_dir, overwrite):
    cfg = _load_config(config_path)
    run_dir = _resolve_run_dir(run_id, Path(runs_dir), overwrite)
    result = build_index(cfg, run_dir)
    click.echo(f"Done. {result['threat_count']} vectors, dim={result['dim']}")


@cli.command("scan-moltbook", hidden=True)
@click.option("--run-id", required=True)
@click.option("--config", "config_path", default=str(_DEFAULT_CONFIG), show_default=True)
@click.option("--runs-dir", type=click.Path(file_okay=False, path_type=Path),
              default=_DEFAULT_DATA_DIR, show_default=True)
@click.option("--nodes-path", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None, help="Optional parquet of nodes to scan; must include node_id, embed_text")
@click.option("--overwrite", is_flag=True)
def cmd_scan(run_id, config_path, runs_dir, nodes_path, overwrite):
    cfg = _load_config(config_path)
    run_dir = _resolve_run_dir(run_id, Path(runs_dir), overwrite)
    result = scan_moltbook(cfg, run_dir, nodes_path=nodes_path)
    click.echo(f"Done. {result['rows']} rows at {result['rows_per_sec']:.1f} rows/sec")


if __name__ == "__main__":
    cli()
