"""Tests for the FastAPI serving layer (CD pillar)."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app, load_models

client = TestClient(app)

MODEL_DIR = Path(__file__).resolve().parents[1] / "Model"
VALID_PAYLOAD = {
    "Type": "M", "Air_temperature": 300.0, "Process_temperature": 310.0,
    "Rotational_speed": 1500, "Torque": 40.0, "Tool_wear": 100,
}


def _models_available() -> bool:
    try:
        load_models()
        return True
    except Exception:
        return False


def test_healthz():
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_version():
    r = client.get("/version")
    assert r.status_code == 200
    assert r.json()["api_version"] == "1.0.0"


def test_predict_rejects_invalid_input():
    """Out-of-range input is rejected (422) without needing the model."""
    bad = dict(VALID_PAYLOAD, Rotational_speed=5000)  # > 3000
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_rejects_missing_field():
    incomplete = {k: v for k, v in VALID_PAYLOAD.items() if k != "Torque"}
    r = client.post("/predict", json=incomplete)
    # 422 from pydantic (missing required field).
    assert r.status_code == 422


def test_predict_valid_payload():
    if not _models_available():
        pytest.skip("Model artifacts unavailable (Git LFS pointers); runs locally.")
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert body["rul_hours"] > 0
    assert body["machine_status"] in {"NORMAL", "WARNING", "CRITICAL"}
