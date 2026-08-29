"""Evaluation metrics used by the VIEW360 paper."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


@dataclass(frozen=True)
class AnomalyMetrics:
    roc_auc: float
    pr_auc: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def anomaly_metrics(
    ground_truth: np.ndarray, anomaly_scores: np.ndarray
) -> AnomalyMetrics:
    """Compute frame-level ROC-AUC and trapezoidal PR-AUC."""

    truth = np.asarray(ground_truth).reshape(-1)
    scores = np.asarray(anomaly_scores).reshape(-1)
    if len(truth) != len(scores):
        raise ValueError(
            f"Ground truth and scores differ in length: {len(truth)} != {len(scores)}"
        )
    if not np.isin(truth, (0, 1)).all():
        raise ValueError("Anomaly ground truth must contain only 0 and 1")
    if not np.isfinite(scores).all():
        raise ValueError("Anomaly scores contain NaN or infinity")

    precision, recall, _ = precision_recall_curve(truth, scores)
    return AnomalyMetrics(
        roc_auc=float(roc_auc_score(truth, scores)),
        pr_auc=float(auc(recall, precision)),
    )


def direction_accuracy(
    direction_ground_truth: np.ndarray,
    direction_predictions: np.ndarray,
    background_label: int = 3,
) -> float:
    """Compute direction accuracy on annotated anomaly frames only."""

    truth = np.asarray(direction_ground_truth).reshape(-1)
    predictions = np.asarray(direction_predictions)
    if predictions.ndim == 2:
        predictions = predictions.argmax(axis=1)
    predictions = predictions.reshape(-1)
    if len(truth) != len(predictions):
        raise ValueError(
            "Direction ground truth and predictions differ in length: "
            f"{len(truth)} != {len(predictions)}"
        )
    evaluated = truth != background_label
    if not evaluated.any():
        raise ValueError("Direction ground truth has no annotated anomaly frames")
    return float(np.mean(predictions[evaluated] == truth[evaluated]))
