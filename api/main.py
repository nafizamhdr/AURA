"""AURA Predictive Maintenance REST API (FastAPI).

Endpoints:
  GET  /healthz  - liveness probe
  GET  /version  - API + model version/metadata
  POST /predict  - RUL + failure-type prediction for one sensor reading

Input validation reuses PreprocessingPipeline.validate_input so the API and
training enforce the same 18-feature contract. Heavy model artifacts are loaded
lazily and cached; if they are unavailable (e.g. unresolved Git LFS pointers),
/predict returns 503 while /healthz and /version keep working.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Src"))

from Preprocessing_pipeline import PreprocessingPipeline  # noqa: E402
from api.schemas import SensorReading, PredictionResponse  # noqa: E402

API_VERSION = "1.0.0"
MODEL_DIR = ROOT / "Model"
PIPELINE_PATH = ROOT / "Pipeline" / "preprocessing_pipeline.pkl"

app = FastAPI(title="AURA Predictive Maintenance API", version=API_VERSION)


def _is_lfs_pointer(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            return fh.read(64).startswith(b"version https://git-lfs")
    except OSError:
        return True


@lru_cache(maxsize=1)
def load_models():
    """Load and cache model artifacts. Raises RuntimeError if unavailable."""
    pkls = ["scaler.pkl", "rul_model.pkl", "failure_model.pkl", "label_encoder.pkl"]
    missing = [p for p in pkls if not (MODEL_DIR / p).exists() or _is_lfs_pointer(MODEL_DIR / p)]
    if missing:
        raise RuntimeError(f"Model artifacts unavailable: {', '.join(missing)}")
    return {
        "scaler": joblib.load(MODEL_DIR / "scaler.pkl"),
        "rul_model": joblib.load(MODEL_DIR / "rul_model.pkl"),
        "failure_model": joblib.load(MODEL_DIR / "failure_model.pkl"),
        "label_encoder": joblib.load(MODEL_DIR / "label_encoder.pkl"),
    }


def _bucket(rul_days: float):
    if rul_days < 7:
        return "CRITICAL", "URGENT", "Schedule maintenance IMMEDIATELY (within 1-2 days)"
    if rul_days < 30:
        return "CRITICAL", "HIGH", "Schedule maintenance within 1-2 weeks"
    if rul_days < 60:
        return "WARNING", "MEDIUM", "Schedule maintenance within 4-8 weeks"
    return "NORMAL", "LOW", "Continue routine monitoring"


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/version")
def version():
    meta_path = MODEL_DIR / "model_metadata.json"
    model_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return {
        "api_version": API_VERSION,
        "model": model_meta.get("best_model"),
        "model_trained_at": model_meta.get("training_date") or model_meta.get("retrained_at"),
        "performance": model_meta.get("performance"),
        "models_loaded": load_models.cache_info().currsize > 0,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading):
    data = {k: v for k, v in reading.model_dump().items() if v is not None}

    # Validate first (no heavy artifacts needed) -> 422 on bad input.
    is_valid, error = PreprocessingPipeline().validate_input(data)
    if not is_valid:
        raise HTTPException(status_code=422, detail=error)

    try:
        models = load_models()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    features = PreprocessingPipeline().transform_single(data)
    scaled = models["scaler"].transform(features)
    rul_hours = float(models["rul_model"].predict(scaled)[0])
    rul_days = rul_hours / 24

    failure_type = failure_confidence = failure_probabilities = None
    if rul_days < 60:
        idx = models["failure_model"].predict(scaled)[0]
        proba = models["failure_model"].predict_proba(scaled)[0]
        failure_type = models["label_encoder"].inverse_transform([idx])[0]
        failure_confidence = float(proba[idx])
        failure_probabilities = {
            cls: float(p) for cls, p in zip(models["label_encoder"].classes_, proba)
        }

    status, priority, action = _bucket(rul_days)
    return PredictionResponse(
        rul_hours=rul_hours, rul_days=rul_days, machine_status=status,
        priority=priority, action=action, failure_type=failure_type,
        failure_confidence=failure_confidence, failure_probabilities=failure_probabilities,
    )
