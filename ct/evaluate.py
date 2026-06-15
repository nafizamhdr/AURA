"""Reusable evaluation functions for Continuous Training (CT).

Covers the three CT quality dimensions:
  - performance  : regression_metrics
  - fairness     : mae_by_group, recall_per_class
  - robustness   : prediction_stability
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, recall_score


def regression_metrics(y_true, y_pred) -> dict:
    """MAE, RMSE, R2 for the RUL regressor (in the same unit as y, i.e. hours)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


def mae_by_group(groups, y_true, y_pred):
    """Mean absolute error per group value.

    Returns (per_group: dict, gap: float) where gap = max(MAE) - min(MAE).
    Used for fairness: a small gap means the model is similarly accurate
    across, e.g., machine types H/M/L.
    """
    frame = pd.DataFrame({
        "group": np.asarray(groups),
        "abs_err": np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)),
    })
    per_group = frame.groupby("group")["abs_err"].mean().to_dict()
    gap = max(per_group.values()) - min(per_group.values())
    return per_group, float(gap)


def recall_per_class(y_true, y_pred, labels) -> dict:
    """Recall for each class label (fairness across failure types)."""
    scores = recall_score(y_true, y_pred, labels=list(labels), average=None, zero_division=0)
    return {label: float(score) for label, score in zip(labels, scores)}


def prediction_stability(model, X, noise_scale: float = 0.01, seed: int = 0) -> float:
    """Mean relative change in predictions when X is perturbed with Gaussian
    noise proportional to each feature's std.

    A small value means the model is robust to minor sensor noise.
    """
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    base = model.predict(X)
    noise = rng.normal(0.0, 1.0, X.shape) * X.std(axis=0) * noise_scale
    perturbed = model.predict(X + noise)
    denom = np.maximum(np.abs(base), 1e-9)
    return float(np.mean(np.abs(perturbed - base) / denom))
