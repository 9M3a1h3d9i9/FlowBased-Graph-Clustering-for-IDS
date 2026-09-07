import numpy as np

from src.evaluation import align_binary_labels, binary_metrics


def test_binary_alignment_is_evaluation_only():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    aligned = align_binary_labels(y_true, y_pred)
    assert np.array_equal(aligned, y_true)


def test_binary_metrics_contains_ids_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    metrics = binary_metrics(y_true, y_pred)
    for key in (
        "precision", "recall", "f1", "specificity", "fpr",
        "balanced_accuracy", "ari", "nmi", "tn", "fp", "fn", "tp",
    ):
        assert key in metrics
