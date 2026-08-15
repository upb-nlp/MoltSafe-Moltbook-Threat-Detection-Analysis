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
    ap.add_argument("--yes", action="store_true", help="Actually cancel; without it, just preview.")
    args = ap.parse_args()

    state = bc.load_state(args.output_dir)
    client = bc.make_client()
    batch = client.batches.retrieve(state["batch_id"])

    print(f"batch_id : {batch.id}")
    print(f"status   : {batch.status}")
    counts = batch.request_counts
    if counts is not None:
        print(f"requests : {counts.completed} completed / {counts.failed} failed / {counts.total} total")

    if batch.status in {"completed", "failed", "expired", "cancelled", "cancelling"}:
        print(f"\nNothing to do — batch is already '{batch.status}'.")
        return 0

    if not args.yes:
        print("\nPreview only. Re-run with --yes to cancel this batch.")
        return 0

    cancelled = client.batches.cancel(batch.id)
    print(f"\ncancel requested -> status: {cancelled.status}")
    print("It may take a few minutes to reach 'cancelled'. Any completed requests are still "
          "billed and fetchable via fetch_results.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
