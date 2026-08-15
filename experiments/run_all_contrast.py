from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from moltbook_poc import repo_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all contrast k-fold experiment configs.")
    parser.add_argument("--config-dir", type=Path, default=repo_paths.path("contrast_config_dir"))
    parser.add_argument("--target-recall", type=float, default=0.8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", dest="overwrite", action="store_true", default=True)
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    args = parser.parse_args()

    configs = sorted(args.config_dir.glob("*.yaml"))
    if not configs:
        raise SystemExit(f"error: no YAML configs found in {args.config_dir}")

    runner = Path(__file__).with_name("run_kfold_contrast.py")
    for config in configs:
        print()
        print(f"=== {config.name} ===", flush=True)
        command = [
            sys.executable,
            str(runner),
            "--config",
            str(config),
            "--target-recall",
            str(args.target_recall),
            "--folds",
            str(args.folds),
            "--seed",
            str(args.seed),
        ]
        if args.overwrite:
            command.append("--overwrite")
        subprocess.run(command, cwd=repo_paths.root(), check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
