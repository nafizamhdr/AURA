"""Unit tests for the production preprocessing pipeline (18-feature contract)."""
import json

import numpy as np
import pytest

from Preprocessing_pipeline import PreprocessingPipeline


def test_transform_single_shape(pipeline, sample_input):
    feats = pipeline.transform_single(sample_input)
    assert feats.shape == (1, 18)


def test_feature_order_matches_features_json(pipeline, root):
    expected = json.loads((root / "Model" / "features.json").read_text())
    assert len(expected) == 18
    assert pipeline.get_feature_names() == expected


def test_engineered_formulas(pipeline):
    data = {
        "Type": "M", "Air_temperature": 300.0, "Process_temperature": 310.0,
        "Rotational_speed": 1500, "Torque": 40.0, "Tool_wear": 100,
    }
    feats = pipeline.transform_single(data)[0]
    idx = {n: i for i, n in enumerate(pipeline.get_feature_names())}
    assert feats[idx["Temp_Difference"]] == pytest.approx(310.0 - 300.0)
    assert feats[idx["Power"]] == pytest.approx(40.0 * 1500 / 9.5488)
    assert feats[idx["Torque_Speed_Ratio"]] == pytest.approx(40.0 / (1500 + 1))


@pytest.mark.parametrize("machine_type,col", [("H", "Type_H"), ("M", "Type_M"), ("L", "Type_L")])
def test_one_hot_type(pipeline, machine_type, col):
    feats = pipeline.transform_single({
        "Type": machine_type, "Air_temperature": 300, "Process_temperature": 310,
        "Rotational_speed": 1500, "Torque": 40, "Tool_wear": 100,
    })[0]
    idx = {n: i for i, n in enumerate(pipeline.get_feature_names())}
    assert feats[idx[col]] == 1
    for other in {"Type_H", "Type_M", "Type_L"} - {col}:
        assert feats[idx[other]] == 0


def test_validate_input_accepts_valid(pipeline, sample_input):
    ok, msg = pipeline.validate_input(sample_input)
    assert ok, msg


@pytest.mark.parametrize("field,value", [
    ("Air_temperature", 500.0),
    ("Air_temperature", 100.0),
    ("Process_temperature", 50.0),
    ("Rotational_speed", 500),
    ("Rotational_speed", 5000),
    ("Torque", -5.0),
    ("Torque", 200.0),
    ("Tool_wear", 400),
])
def test_validate_input_rejects_out_of_range(pipeline, sample_input, field, value):
    bad = dict(sample_input)
    bad[field] = value
    ok, msg = pipeline.validate_input(bad)
    assert not ok
    assert field in msg


def test_validate_input_rejects_bad_type(pipeline, sample_input):
    bad = dict(sample_input)
    bad["Type"] = "X"
    ok, _ = pipeline.validate_input(bad)
    assert not ok


def test_validate_input_rejects_missing_field(pipeline, sample_input):
    bad = dict(sample_input)
    del bad["Torque"]
    ok, msg = pipeline.validate_input(bad)
    assert not ok
    assert "Torque" in msg


def test_save_load_roundtrip(pipeline, sample_input, tmp_path):
    before = pipeline.transform_single(sample_input)
    path = tmp_path / "pipeline.pkl"
    pipeline.save(str(path))
    loaded = PreprocessingPipeline.load(str(path))
    after = loaded.transform_single(sample_input)
    assert np.allclose(before, after)
