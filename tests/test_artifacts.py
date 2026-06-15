"""Training verification (static): artifact integrity, feature contract,
end-to-end inference, and a performance regression guard."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from Preprocessing_pipeline import PreprocessingPipeline

ARTIFACTS = [
    "Model/rul_model.pkl",
    "Model/scaler.pkl",
    "Model/failure_model.pkl",
    "Model/label_encoder.pkl",
    "Model/features.json",
    "Model/model_metadata.json",
    "Pipeline/preprocessing_pipeline.pkl",
]

# .pkl artifacts are tracked via Git LFS. In environments where the LFS objects
# are not materialized (e.g. CI without the LFS blobs available), the files on
# disk are small pointer stubs ("version https://git-lfs..."). Loading those
# would fail, so we skip the model-loading tests there instead of erroring.
PKL_ARTIFACTS = [
    "Model/scaler.pkl",
    "Model/rul_model.pkl",
    "Model/failure_model.pkl",
    "Model/label_encoder.pkl",
]


def _is_lfs_pointer(path):
    try:
        with open(path, "rb") as fh:
            return fh.read(64).startswith(b"version https://git-lfs")
    except OSError:
        return True


@pytest.mark.parametrize("rel", ARTIFACTS)
def test_artifact_exists(root, rel):
    assert (root / rel).exists(), f"Missing artifact: {rel}"


@pytest.fixture(scope="module")
def models(root):
    pointers = [rel for rel in PKL_ARTIFACTS if _is_lfs_pointer(root / rel)]
    if pointers:
        pytest.skip(
            "Model artifacts are unresolved Git LFS pointers "
            f"({', '.join(pointers)}). Pull the LFS objects to run model-loading "
            "tests (they run in full locally)."
        )
    return {
        "scaler": joblib.load(root / "Model" / "scaler.pkl"),
        "rul": joblib.load(root / "Model" / "rul_model.pkl"),
        "failure": joblib.load(root / "Model" / "failure_model.pkl"),
        "le": joblib.load(root / "Model" / "label_encoder.pkl"),
        "meta": json.loads((root / "Model" / "model_metadata.json").read_text()),
    }


def test_feature_contract(root, models):
    feats = json.loads((root / "Model" / "features.json").read_text())
    assert len(feats) == 18
    assert models["scaler"].n_features_in_ == 18
    assert PreprocessingPipeline().get_feature_names() == feats


def test_end_to_end_inference(root, models):
    pipe = PreprocessingPipeline()
    row = pd.read_csv(root / "Data" / "predictive_maintenance.csv", nrows=1).iloc[0].to_dict()
    feats = pipe.transform_single(row)
    scaled = models["scaler"].transform(feats)

    rul_hours = float(models["rul"].predict(scaled)[0])
    assert np.isfinite(rul_hours)

    # Two-stage logic: failure classifier only runs when RUL < 60 days.
    if rul_hours / 24 < 60:
        idx = models["failure"].predict(scaled)[0]
        cls = models["le"].inverse_transform([idx])[0]
        assert cls in list(models["le"].classes_)


def test_performance_regression_guard(models):
    perf = models["meta"]["performance"]
    # Guard against silent degradation versus the recorded baseline.
    assert perf["test_mae_days"] <= 10.0
    assert perf["test_r2"] >= 0.95
