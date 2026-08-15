
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import batch_common as bc


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, required=True, help="Dir holding batch_state.json.")
    args = ap.parse_args()

    state = bc.load_state(args.output_dir)
    client = bc.make_client()
    batch = client.batches.retrieve(state["batch_id"])

    counts = batch.request_counts
    print(f"batch_id : {batch.id}")
    print(f"status   : {batch.status}")
    if counts is not None:
        print(f"requests : {counts.completed} completed / {counts.failed} failed / {counts.total} total")
    if getattr(batch, "output_file_id", None):
        print(f"output_file_id: {batch.output_file_id}")
    if getattr(batch, "error_file_id", None):
        print(f"error_file_id : {batch.error_file_id}")
    if batch.status == "completed":
        print("\nReady — run fetch_results.py --output-dir", str(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
