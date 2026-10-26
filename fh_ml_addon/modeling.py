from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score,
)


def make_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=120,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        max_bins=127,
        early_stopping=False,
        random_state=42,
    )


def fit_model(x: pd.DataFrame, y: pd.Series) -> HistGradientBoostingClassifier:
    labels = y.to_numpy(dtype=int)
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2:
        raise ValueError(f"Training labels need both classes; observed counts: {dict(zip(classes.tolist(), counts.tolist()))}")
    n = len(labels)
    freq = {int(c): int(k) for c, k in zip(classes, counts)}
    weights = np.asarray([n / (len(classes) * freq[int(v)]) for v in labels], dtype=float)
    model = make_model()
    model.fit(x, labels, sample_weight=weights)
    return model


def best_f1_threshold(y_true: pd.Series | np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    y, scores = y[finite], scores[finite]
    if not len(y):
        return 0.5, 0.0
    if len(np.unique(y)) < 2 or len(np.unique(scores)) < 2:
        candidates = [float(np.nanmedian(scores)) if len(scores) else 0.5, 0.5]
        scored = [(float(t), f1_score(y, scores >= t, zero_division=0)) for t in candidates]
        best = max(scored, key=lambda q: (q[1], q[0]))
        return best
    # Use the PR curve so thresholds are evaluated on validation data only.
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y, scores)
    if len(thresholds) == 0:
        return 0.5, float(f1_score(y, scores >= 0.5, zero_division=0))
    denom = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denom, out=np.zeros_like(denom), where=denom > 0)
    max_f1 = float(np.max(f1))
    best_indexes = np.flatnonzero(np.isclose(f1, max_f1))
    # Conservative tie-break: choose the larger threshold among equal-F1 choices.
    ix = int(best_indexes[-1])
    return float(thresholds[ix]), max_f1


def metric_dict(y_true: pd.Series | np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    keep = np.isfinite(s)
    y, s = y[keep], s[keep]
    if not len(y):
        return {"n": 0, "positive_rate": None, "average_precision": None,
                "precision": None, "recall": None, "f1": None, "balanced_accuracy": None,
                "threshold": float(threshold)}
    predicted = s >= float(threshold)
    two_classes = len(np.unique(y)) == 2
    return {
        "n": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "average_precision": float(average_precision_score(y, s)) if two_classes else None,
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)) if two_classes else None,
        "threshold": float(threshold),
    }


def persistence_scores(features: pd.DataFrame) -> np.ndarray:
    """Simple non-ML baseline: number of qualifying C-anomaly days in the prior 7 days."""
    column = "C_active_days_7d"
    if column not in features:
        return np.zeros(len(features), dtype=float)
    return pd.to_numeric(features[column], errors="coerce").fillna(0).to_numpy(dtype=float)
