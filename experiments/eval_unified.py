import hashlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from moltbook_poc import repo_paths

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = repo_paths.root()

POSITIVE_SEVERITIES = {"3", "4", "5"} 

CANONICAL_TEST_FINGERPRINT = "20aac09c21fe7ecac9ea1fccc57a3c49b1093f8db80e89769dec83860bf3aa69"


def test_set_fingerprint(node_ids: pd.Series) -> str:
    return hashlib.sha256("\n".join(sorted(node_ids)).encode()).hexdigest()


def compute_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    n = int(len(y))
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0

    return dict(
        n_flagged=tp + fp,
        flag_rate=(tp + fp) / n if n else float("nan"),
        tp=tp,
        fp=fp,
        fn=fn,
        precision=prec,
        recall=rec,
    )


def compute_ranking_metrics(y: np.ndarray, score: np.ndarray) -> dict:
    if len(np.unique(y)) < 2:
        return dict(average_precision=float("nan"), roc_auc=float("nan"))
    return dict(
        average_precision=float(average_precision_score(y, score)),
        roc_auc=float(roc_auc_score(y, score)),
    )
