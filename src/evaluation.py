"""External and internal clustering metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
    silhouette_score,
)


def align_binary_labels(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Align a binary clustering orientation for offline evaluation only."""
    y_pred = np.asarray(y_pred).astype(int)
    if f1_score(y_true, y_pred, zero_division=0) >= f1_score(y_true, 1 - y_pred, zero_division=0):
        return y_pred
    return 1 - y_pred


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute IDS-oriented binary metrics."""
    y_pred = align_binary_labels(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "fpr": float(1.0 - specificity),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute Euclidean silhouette score in the supplied feature space."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(silhouette_score(X, labels))
