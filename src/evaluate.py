"""evaluation metrics for fraud detection."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
)


@dataclass
class Metrics:
    pr_auc: float
    roc_auc: float
    recall_at_p50: float
    recall_at_p90: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def recall_at_precision(
    y_true: np.ndarray, y_score: np.ndarray, target_precision: float
) -> float:
    """highest recall achievable while precision >= target_precision.

    returns 0.0 if the target precision is never reached.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    mask = precision >= target_precision
    if not mask.any():
        return 0.0
    return float(recall[mask].max())


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Metrics:
    """compute all baseline metrics."""
    return Metrics(
        pr_auc=float(average_precision_score(y_true, y_score)),
        roc_auc=float(roc_auc_score(y_true, y_score)),
        recall_at_p50=recall_at_precision(y_true, y_score, 0.5),
        recall_at_p90=recall_at_precision(y_true, y_score, 0.9),
    )