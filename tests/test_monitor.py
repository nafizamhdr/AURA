"""Tests for data-drift monitoring (CD pillar)."""
import numpy as np
import pandas as pd

from cd.monitor import psi, detect_drift

FEATURES = ["Air_temperature", "Torque", "Rotational_speed"]


def _frame(rng, shift=0.0, n=2000):
    return pd.DataFrame({
        "Air_temperature": rng.normal(300 + shift, 2, n),
        "Torque": rng.normal(40 + shift, 5, n),
        "Rotational_speed": rng.normal(1500 + shift * 50, 100, n),
    })


def test_psi_near_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    cur = rng.normal(0, 1, 5000)
    assert psi(ref, cur) < 0.1


def test_psi_high_for_shifted_distribution():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    cur = rng.normal(3, 1, 5000)  # large shift
    assert psi(ref, cur) > 0.25


def test_detect_drift_flags_no_drift():
    rng = np.random.default_rng(1)
    result = detect_drift(_frame(rng), _frame(rng), FEATURES)
    assert result["drift_detected"] is False
    assert result["drifted_features"] == []


def test_detect_drift_flags_drift():
    rng = np.random.default_rng(2)
    reference = _frame(rng, shift=0.0)
    current = _frame(rng, shift=8.0)  # clearly drifted
    result = detect_drift(reference, current, FEATURES)
    assert result["drift_detected"] is True
    assert len(result["drifted_features"]) >= 1
