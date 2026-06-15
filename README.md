---
title: AURA Predictive Maintenance
emoji: 🔧
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.36.0
app_file: app.py
pinned: false
license: mit
---

# AURA — Predictive Maintenance Demo

Interactive Streamlit demo that predicts **Remaining Useful Life (RUL)** for industrial machines from sensor readings, and classifies the likely **failure type** when RUL drops below 60 days.

## What it does

Given a single sensor reading (air/process temperature, rotational speed, torque, tool wear, machine type), the app:

1. Validates input ranges.
2. Builds an 18-feature vector (5 raw sensor + 5 engineered + 3 datetime + 2 temporal + 3 one-hot type).
3. Predicts RUL in hours using a Random Forest regressor (test MAE ≈ 6.8 days, R² ≈ 0.986).
4. If RUL < 60 days, runs an XGBoost classifier over 5 failure modes (Heat Dissipation, Overstrain, Power, Random, Tool Wear).
5. Buckets the result into NORMAL / WARNING / CRITICAL with a recommended action.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open <http://localhost:8501>. Use the **preset buttons** in the sidebar to load one of 5 scenarios (NORMAL → CRITICAL) without filling the form manually.

## Project layout

- `app.py` — Streamlit entrypoint.
- `Src/Preprocessing_pipeline.py` — production preprocessing class (single source of truth for the 18-feature contract).
- `Src/{preprocessing,FE,Model,Test}.ipynb` — training pipeline notebooks (run in order).
- `Model/` — trained artifacts (`rul_model.pkl`, `scaler.pkl`, `failure_model.pkl`, `label_encoder.pkl`, `features.json`, `model_metadata.json`).
- `Pipeline/preprocessing_pipeline.pkl` — pickled preprocessor.
- `Data/` — source CSVs (not deployed).
- `tests/` — pytest suite (unit, data validation, training verification, fairness, robustness, API, registry, monitoring).
- `ct/` — Continuous Training: `features.py` (reproducible feature engineering), `evaluate.py` (metrics), `train.py` (retrain + quality gate).
- `cd/` — Continuous Deployment: `registry.py` (file-based model registry), `monitor.py` (PSI drift detection).
- `api/` — FastAPI serving (`/predict`, `/healthz`, `/version`).
- `Dockerfile` — container image for the API.
- `.github/workflows/` — `ci.yml`, `ct.yml`, `cd.yml`.

## MLOps Pipeline (CI / CT / CD)

The project is automated with three GitHub Actions workflows:

| Workflow | Trigger | What it does |
| --- | --- | --- |
| **CI** (`ci.yml`) | every push / PR | Runs the full pytest suite — unit tests, data validation (Pandera), training verification, fairness & robustness. |
| **CT** (`ct.yml`) | manual / weekly cron | Recomputes features from the raw CSV, retrains, and applies a quality gate (MAE / R² / fairness). Does **not** promote unless the gate passes. |
| **CD** (`cd.yml`) | manual / version tag | Runs API, registry and monitoring tests, then builds the Docker image. |

### Run the tests

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

### Run the API

```bash
uvicorn api.main:app --reload
# then open http://localhost:8000/docs  (interactive Swagger UI)
```

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"Type":"M","Air_temperature":300,"Process_temperature":310,"Rotational_speed":1500,"Torque":40,"Tool_wear":100}'
```

### Continuous training (dry-run)

```bash
python -m ct.train            # train + evaluate + quality gate, no promotion
python -m ct.train --promote  # write artifacts only if the gate passes
```

## Note on cold start

This Space runs on HuggingFace's free tier and may sleep after extended inactivity. The first request after sleep can take ~30–60 seconds while the container wakes up.
