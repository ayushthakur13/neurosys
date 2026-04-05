from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


@dataclass
class BinaryMetrics:
    precision: float
    recall: float
    f1: float
    roc_auc: float
    tn: int
    fp: int
    fn: int
    tp: int


def threshold_by_percentile(scores: np.ndarray, percentile: float) -> float:
    return float(np.percentile(scores, percentile))


def threshold_by_f1_optimization(y_true: np.ndarray, scores: np.ndarray, num_steps: int = 200) -> tuple[float, float]:
    """Select a threshold that maximizes F1 on a validation set."""
    if len(scores) == 0:
        return 0.0, 0.0

    candidates = np.unique(scores)
    if len(candidates) > num_steps:
        quantiles = np.linspace(0.0, 1.0, num_steps)
        candidates = np.unique(np.quantile(scores, quantiles))

    best_threshold = float(candidates[0])
    best_f1 = -1.0
    for threshold in candidates:
        predictions = (scores >= threshold).astype(int)
        score = f1_score(y_true, predictions, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)

    return best_threshold, best_f1


def compute_binary_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> BinaryMetrics:
    y_pred = (scores >= threshold).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else 0.0
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return BinaryMetrics(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        roc_auc=float(roc_auc),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )
