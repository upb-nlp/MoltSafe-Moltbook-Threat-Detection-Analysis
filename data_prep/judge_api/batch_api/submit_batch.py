from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

import batch_common as bc
from batch_common import ne  


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=ne.DEFAULT_INPUT_CSV, help="Sample-index CSV to read.")
    ap.add_argument("--prompt", type=Path, default=ne.DEFAULT_PROMPT, help="Prompt file (your instructions).")
    ap.add_argument("--output-dir", type=Path, required=True, help="Where to write batch state + results.")
    ap.add_argument("--model", default=ne.MODEL, help=f"Model id (default: {ne.MODEL}).")
    ap.add_argument("--effort", default=ne.REASONING_EFFORT, help=f"Reasoning effort (default: {ne.REASONING_EFFORT}).")
    ap.add_argument("--dry-run", action="store_true", help="Build batch_input.jsonl and stop; no upload.")
    ap.add_argument("--force", action="store_true", help="Resubmit even if batch_state.json exists (bills a new batch).")
    args = ap.parse_args()

    if not args.input.exists():
        print(f"ERROR: input CSV not found: {args.input}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if bc.state_path(args.output_dir).exists() and not args.force:
        print(f"ERROR: {bc.state_path(args.output_dir)} already exists — a batch was already "
              f"submitted here.\n  Re-run with --force to submit a NEW (separately billed) batch, "
              f"or pick a different --output-dir.", file=sys.stderr)
        return 1

    prompt_text = ne.load_prompt(args.prompt)
    sample = pd.read_csv(args.input).sort_values("example_num").reset_index(drop=True)

    batch_input = args.output_dir / "batch_input.jsonl"
    with batch_input.open("w", encoding="utf-8") as fh:
        for _, row in sample.iterrows():
            line = bc.to_batch_line(row, prompt_text, args.model, args.effort)
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(f"built {len(sample)} requests -> {batch_input}")

    if args.dry_run:
        first = json.loads(batch_input.read_text(encoding="utf-8").splitlines()[0])
        print("\n--- first request line (dry run, nothing uploaded) ---")
        print(json.dumps(first, indent=2, ensure_ascii=False))
        return 0

    client = bc.make_client()
    uploaded = client.files.create(file=batch_input.open("rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint=bc.BATCH_ENDPOINT,
        completion_window=bc.COMPLETION_WINDOW,
    )

    bc.save_state(args.output_dir, {
        "batch_id": batch.id,
        "input_file_id": uploaded.id,
        "endpoint": bc.BATCH_ENDPOINT,
        "completion_window": bc.COMPLETION_WINDOW,
        "input_csv": str(args.input),
        "prompt_file": str(args.prompt),
        "model": args.model,
        "effort": args.effort,
        "submitted_at": ne.utc_now_iso(),
    })

    print(f"\nsubmitted batch {batch.id} (status: {batch.status})")
    print(f"state -> {bc.state_path(args.output_dir)}")
    print("Next: check_status.py --output-dir <dir>, then fetch_results.py when completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
