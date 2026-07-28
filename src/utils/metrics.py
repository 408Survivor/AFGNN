"""
Evaluation metrics for depression classification.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5):
    """Compute classification metrics.

    Args:
        y_true: ground-truth labels, shape (N,).
        y_score: predicted probabilities (after sigmoid), shape (N,).
        threshold: threshold for binary prediction.

    Returns:
        dict with accuracy, precision, recall, f1, auc.
    """
    y_pred = (y_score >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    try:
        metrics["auc"] = roc_auc_score(y_true, y_score)
    except ValueError:
        # In case only one class is present
        metrics["auc"] = float("nan")

    return metrics


def find_best_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: str = "f1",
    thresholds: np.ndarray = None,
):
    """Search for the threshold that maximizes a given metric on the validation set.

    This is useful for imbalanced binary classification: we use AUC (threshold-agnostic)
    to select the best model/checkpoint, and then choose the operating threshold that
    yields the best F1 (or other metric) on the validation set.

    Args:
        y_true: ground-truth labels, shape (N,).
        y_score: predicted probabilities, shape (N,).
        metric: metric to optimize. Currently only "f1" is supported.
        thresholds: array of thresholds to try. Defaults to 0.01 ~ 0.99 with step 0.01.

    Returns:
        best_threshold: threshold that gives the best metric.
        best_score: best metric value achieved.
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)

    if metric == "f1":
        metric_fn = lambda y_t, y_p: f1_score(y_t, y_p, zero_division=0)
    else:
        raise ValueError(f"Unsupported threshold search metric: {metric}")

    best_threshold = 0.5
    best_score = -float("inf")

    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        score = metric_fn(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_threshold = float(thr)

    return best_threshold, best_score
