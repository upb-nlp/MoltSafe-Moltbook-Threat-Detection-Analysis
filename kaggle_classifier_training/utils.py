from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


def argmax_pred(scores: np.ndarray) -> np.ndarray:

    p_unsafe = np.asarray(scores, dtype=np.float64)
    p_safe = 1.0 - p_unsafe
    return (p_unsafe > p_safe).astype(np.int64)


def compute_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    y = np.asarray(y)
    pred = np.asarray(pred)
    n = int(len(y))
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return dict(
        n=n, n_pos=int((y == 1).sum()),
        n_flagged=tp + fp, flag_rate=(tp + fp) / n if n else float("nan"),
        tp=tp, fp=fp, fn=fn, tn=tn,
        precision=prec, recall=rec,
        accuracy=(tp + tn) / n if n else float("nan"),
        f1=f1,
    )


def compute_ranking_metrics(y: np.ndarray, score: np.ndarray) -> dict:
    y = np.asarray(y)
    score = np.asarray(score)
    if len(np.unique(y)) < 2:
        return dict(pr_auc=float("nan"), average_precision=float("nan"), roc_auc=float("nan"))
    prec, rec, _ = precision_recall_curve(y, score)
    return dict(
        pr_auc=float(auc(rec, prec)),
        average_precision=float(average_precision_score(y, score)),
        roc_auc=float(roc_auc_score(y, score)),
    )
