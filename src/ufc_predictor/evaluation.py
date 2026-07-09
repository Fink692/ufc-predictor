from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score


def classification_metrics(y_true, probabilities, threshold: float = 0.5) -> dict[str, float | None]:
    y = np.asarray(y_true, dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    clipped = np.clip(probs, 1e-6, 1.0 - 1e-6)
    predictions = (probs >= threshold).astype(int)
    metrics: dict[str, float | None] = {
        "row_count": int(len(y)),
        "log_loss": float(log_loss(y, np.column_stack([1.0 - clipped, clipped]), labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, clipped)),
        "accuracy": float(accuracy_score(y, predictions)),
    }
    if len(set(y.tolist())) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y, probs))
    else:
        metrics["roc_auc"] = None
    return metrics
