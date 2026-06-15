"""Smoke training: verify the training stack runs end-to-end and is reproducible.

Trains a tiny model on a small subset so CI stays fast; this checks that the
feature pipeline + estimator can fit and predict without errors.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from Preprocessing_pipeline import PreprocessingPipeline


def _features_and_target(root, nrows):
    df = pd.read_csv(root / "Data" / "predictive_maintenance.csv", nrows=nrows)
    X = PreprocessingPipeline().transform_batch(df.to_dict("records"))
    y = df["RUL_hours"].to_numpy()
    return X, y


def test_smoke_training_runs(root):
    X, y = _features_and_target(root, nrows=2000)
    assert X.shape == (2000, 18)

    model = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=1)
    model.fit(X, y)
    preds = model.predict(X[:5])

    assert preds.shape == (5,)
    assert np.all(np.isfinite(preds))


def test_training_is_reproducible(root):
    X, y = _features_and_target(root, nrows=1000)
    m1 = RandomForestRegressor(n_estimators=8, random_state=0, n_jobs=1).fit(X, y)
    m2 = RandomForestRegressor(n_estimators=8, random_state=0, n_jobs=1).fit(X, y)
    assert np.allclose(m1.predict(X[:20]), m2.predict(X[:20]))
