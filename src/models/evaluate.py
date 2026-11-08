"""Evaluation helpers shared by training, retraining and monitoring."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


def evaluate(y_true, p_pred) -> dict[str, float]:
    """AUC (main), log-loss and accuracy.  AUC is NaN when only one class is present."""
    y = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(p_pred, dtype=float), 1e-6, 1 - 1e-6)
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    return {
        "auc": auc,
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "accuracy": float(accuracy_score(y, p >= 0.5)),
    }


def time_split(df, valid_fraction: float):
    """Chronological split: the LAST ``valid_fraction`` of rows (by time) is validation."""
    df = df.sort_values(["ts", "order_id"], kind="mergesort")
    cut = int(len(df) * (1 - valid_fraction))
    return df.iloc[:cut], df.iloc[cut:]
