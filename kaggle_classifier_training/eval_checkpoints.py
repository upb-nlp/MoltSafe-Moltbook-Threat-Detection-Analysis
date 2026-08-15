from __future__ import annotations

import argparse
import json
import re
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch

import utils
from train_head import (
    ON_KAGGLE, REPO_ROOT, DEFAULT_CONFIG,
    Config, SafetyClassifier, load_data, prepare_texts, predict_proba,
)

EPOCH_RE = re.compile(r"head_epoch(\d+)\.pt$")

TABLE_COLS = ["epoch", "precision", "recall", "accuracy", "f1",
              "tp", "fp", "fn", "tn", "n_flagged", "flag_rate", "train_loss"]
COUNT_COLS = {"epoch", "tp", "fp", "fn", "tn", "n_flagged"}


def find_checkpoints(ckpt_dir: Path) -> list[tuple[int, Path]]:
    found = []
    for p in ckpt_dir.glob("head_epoch*.pt"):
        m = EPOCH_RE.search(p.name)
        if m:
            found.append((int(m.group(1)), p))
    if not found:
        raise FileNotFoundError(
            f"no head_epoch*.pt in {ckpt_dir} -- run train_head.py first, or point "
            f"--ckpt-dir at the run's checkpoints/ directory"
        )
    return sorted(found)


def fmt(col: str, v) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "-"
    if col in COUNT_COLS:
        return str(int(v))
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def print_table(df: pd.DataFrame) -> None:
    widths = {c: max(len(c), max(len(fmt(c, v)) for v in df[c])) for c in df.columns}
    print("  ".join(c.rjust(widths[c]) for c in df.columns))
    print("  ".join("-" * widths[c] for c in df.columns))
    for _, r in df.iterrows():
        print("  ".join(fmt(c, r[c]).rjust(widths[c]) for c in df.columns))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--ckpt-dir", type=Path, default=None,
                    help="directory of head_epoch*.pt (default: <run>/checkpoints)")
    ap.add_argument("--out", type=Path,
                    default=Path("/kaggle/working/head_clf_stratified") if ON_KAGGLE
                    else REPO_ROOT / "runs_head_clf",
                    help="training output root, used to derive the default --ckpt-dir")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="where to write scores + metrics (default: <run>/checkpoint_eval)")
    ap.add_argument("--variant", choices=["noprefix", "instruct"], default=None)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--last-only", action="store_true",
                    help="score only the highest-numbered checkpoint (the final epoch), an "
                         "a-priori rule that avoids selecting a checkpoint on the test set")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    overrides = {k: v for k, v in (("variant", args.variant), ("hidden", args.hidden))
                 if v is not None}
    if overrides:
        cfg = replace(cfg, **overrides)

    run_dir = args.out / f"{cfg.variant}_h{cfg.hidden}"
    ckpt_dir = args.ckpt_dir or (run_dir / "checkpoints")
    out_dir = args.out_dir or (run_dir / "checkpoint_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = find_checkpoints(ckpt_dir)
    n_found = len(checkpoints)
    if args.last_only:
        checkpoints = checkpoints[-1:]     
    print(f"config={args.config}")
    print(f"checkpoints={ckpt_dir}  ({n_found} found, {len(checkpoints)} scored"
          f"{' -- --last-only' if args.last_only else ''})")
    print(f"out={out_dir}\n")


    print("DATA")
    test_raw, y_test, test_ids = load_data("test", cfg)
    test_texts = prepare_texts(test_raw, cfg)   
    if test_ids is None:
        raise FileNotFoundError(
            "data/test_node_ids.txt is missing -- it is required so the score files can be "
            "joined by eval_unified.py. Re-upload the fixed data/ bundle."
        )

    print("\nMODEL")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SafetyClassifier(cfg).to(device)
    if cfg.fp16 and device.type == "cuda":
        model.encoder.half()
    print(f"  {cfg.model_name} on {device}  max_seq_length={cfg.max_seq_length}")

    print(f"\nEVALUATE  ({len(checkpoints)} checkpoints x {len(test_texts)} texts, "
          f"full inference each)")
    rows = []
    for epoch, path in checkpoints:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        if ckpt.get("hidden") != cfg.hidden or ckpt.get("dim") != model.dim:
            raise ValueError(
                f"{path.name}: architecture mismatch (hidden={ckpt.get('hidden')}, "
                f"dim={ckpt.get('dim')}) vs the model built from {args.config} "
                f"(hidden={cfg.hidden}, dim={model.dim})"
            )
        model.head.load_state_dict(ckpt["state_dict"])

        scores = predict_proba(model, test_texts, device, f"epoch{epoch:02d}")
        assert ((scores >= 0) & (scores <= 1)).all(), f"epoch {epoch}: scores are not probabilities"

        scores_path = out_dir / f"test_scores_epoch{epoch:02d}.parquet"
        pd.DataFrame({"node_id": test_ids, "label": y_test, "score": scores}).to_parquet(
            scores_path, index=False)

        m = utils.compute_metrics(y_test, utils.argmax_pred(scores))
        assert m["tp"] + m["fp"] + m["fn"] + m["tn"] == len(y_test), \
            f"epoch {epoch}: confusion matrix does not cover all {len(y_test)} rows"
        assert m["tp"] + m["fn"] == int(y_test.sum()), \
            f"epoch {epoch}: positives lost ({m['tp'] + m['fn']} != {int(y_test.sum())})"

        m.update(epoch=epoch, train_loss=ckpt.get("train_loss"),
                 checkpoint=path.name, scores_file=scores_path.name)
        rows.append(m)
        print(f"  epoch {epoch:2d}  P {m['precision']:.4f}  R {m['recall']:.4f}  "
              f"acc {m['accuracy']:.4f}  F1 {m['f1']:.4f}  flagged {m['n_flagged']:4d}")

    df = pd.DataFrame(rows)
    csv_path = out_dir / "checkpoint_metrics.csv"
    df.to_csv(csv_path, index=False)

    n, n_pos = len(y_test), int(y_test.sum())
    print(f"\nCHECKPOINT RESULTS  ({n} test nodes, {n_pos} unsafe, "
          f"argmax rule: p_unsafe > p_safe)")
    print_table(df[TABLE_COLS].sort_values("f1", ascending=False).reset_index(drop=True))

    print(f"\n  all-negative baseline: accuracy {1 - n_pos / n:.4f} at {n_pos / n:.1%} "
          f"prevalence -- accuracy below that line is worse than predicting nothing.")
    print("  Sorted by F1 to match eval_unified.py's ordering; recall is in the table if that")
    print("  is what you are optimising for. Nothing here has been promoted.")
    if args.last_only:
        epoch = int(df["epoch"].iloc[0])
        print(f"  --last-only: epoch {epoch} was fixed before evaluation (the final epoch), so")
        print(f"     this score carries no max-over-N selection bias. {n_found - 1} unscored")
        print("     checkpoint(s) remain on disk; scoring them re-introduces that bias.")
    else:
        print(f"  !! These {n} rows are also the selection set -- a checkpoint picked from this")
        print("     table carries selection bias that the contrast runs do not.")

    (out_dir / "eval_manifest.json").write_text(json.dumps({
        "config": cfg.to_json(),
        "ckpt_dir": str(ckpt_dir),
        "n_test": n, "n_test_unsafe": n_pos,
        "decision_rule": "argmax (p_unsafe > p_safe)",
        "metrics_source": "utils.py, vendored from scripts/evaluation/eval_unified.py",
        "held_out": False,
        "n_checkpoints": len(rows),
        "last_only": args.last_only,
        "n_checkpoints_found": n_found,
    }, indent=2), encoding="utf-8")

    print(f"\nwrote {csv_path}")
    print(f"wrote {out_dir / 'eval_manifest.json'}")
    print(f"wrote {len(rows)} x test_scores_epoch*.parquet -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
