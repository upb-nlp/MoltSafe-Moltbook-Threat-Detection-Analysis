from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    y = np.asarray(y)
    pred = np.asarray(pred)
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
    y = np.asarray(y)
    score = np.asarray(score)
    if len(np.unique(y)) < 2:
        return dict(average_precision=float("nan"), roc_auc=float("nan"))
    return dict(
        average_precision=float(average_precision_score(y, score)),
        roc_auc=float(roc_auc_score(y, score)),
    )
