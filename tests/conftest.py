"""Shared pytest fixtures and path setup for AURA CI tests."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          # for `import ct...`
sys.path.insert(0, str(ROOT / "Src"))  # for `import Preprocessing_pipeline`

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def root():
    return ROOT


@pytest.fixture(scope="session")
def pipeline():
    from Preprocessing_pipeline import PreprocessingPipeline
    return PreprocessingPipeline()


@pytest.fixture(scope="session")
def sample_input():
    """A single valid sensor reading (mirrors Preprocessing_pipeline demo)."""
    return {
        "datetime": "2025-01-20 14:30:00",
        "Type": "M",
        "Air_temperature": 300.0,
        "Process_temperature": 310.0,
        "Rotational_speed": 1480,
        "Torque": 42.0,
        "Tool_wear": 150,
        "machine_age_hours": 15000,
        "hours_since_last": 8,
        "Temp_Rate_of_Change": 0.15,
        "RPM_Variance": 35.0,
    }


@pytest.fixture(scope="session")
def ct_regression():
    """Small RUL regressor trained on a data subset (CT fairness/robustness).

    Independent of the shipped 142 MB LFS model so it runs anywhere, fast.
    """
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from Preprocessing_pipeline import PreprocessingPipeline

    df = pd.read_csv(ROOT / "Data" / "predictive_maintenance.csv", nrows=6000)
    X = PreprocessingPipeline().transform_batch(df.to_dict("records"))
    y = df["RUL_hours"].to_numpy()
    idx = np.arange(len(df))
    X_tr, X_te, y_tr, y_te, i_tr, i_te = train_test_split(
        X, y, idx, test_size=0.25, random_state=42
    )
    model = RandomForestRegressor(n_estimators=40, random_state=42, n_jobs=1).fit(X_tr, y_tr)
    return {
        "model": model,
        "X_test": X_te,
        "y_test": y_te,
        "pred": model.predict(X_te),
        "df_test": df.iloc[i_te].reset_index(drop=True),
    }


@pytest.fixture(scope="session")
def ct_classification():
    """Small failure-type classifier trained on the (rare) failure rows.

    The dataset has only ~250 failure rows across 5 classes, so this is a
    loose, mechanism-level fixture rather than a high-accuracy model.
    """
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from Preprocessing_pipeline import PreprocessingPipeline

    df = pd.read_csv(ROOT / "Data" / "predictive_maintenance.csv")
    fail = df[df["Failure_Type"] != "No Failure"].reset_index(drop=True)
    X = PreprocessingPipeline().transform_batch(fail.to_dict("records"))
    y = fail["Failure_Type"].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=60, random_state=42, n_jobs=1).fit(X_tr, y_tr)
    return {"model": model, "X_test": X_te, "y_test": y_te, "pred": model.predict(X_te)}
