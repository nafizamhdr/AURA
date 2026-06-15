"""Data-drift monitoring (CD pillar).

Computes the Population Stability Index (PSI) per feature between a reference
distribution (e.g. training data) and a current batch (e.g. recent live
traffic). High PSI signals drift, which closes the loop back to CT by
triggering a retrain.

PSI interpretation (industry rule of thumb):
  < 0.10  : no significant change
  0.10-0.25 : moderate shift (investigate)
  > 0.25  : significant drift (retrain)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DRIFT_THRESHOLD = 0.25


def psi(reference, current, bins: int = 10) -> float:
    """Population Stability Index between two 1-D distributions."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    # Bin edges from the reference quantiles (robust to scale).
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 3:  # near-constant feature -> no meaningful PSI
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_pct = np.histogram(reference, bins=edges)[0] / len(reference)
    cur_pct = np.histogram(current, bins=edges)[0] / len(current)

    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def detect_drift(reference: pd.DataFrame, current: pd.DataFrame, features,
                 threshold: float = DRIFT_THRESHOLD) -> dict:
    """Return per-feature PSI plus an overall drift decision."""
    scores = {f: psi(reference[f], current[f]) for f in features}
    drifted = [f for f, v in scores.items() if v > threshold]
    return {
        "psi": scores,
        "drifted_features": drifted,
        "drift_detected": bool(drifted),
        "threshold": threshold,
    }
