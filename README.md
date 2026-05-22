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

## Note on cold start

This Space runs on HuggingFace's free tier and may sleep after extended inactivity. The first request after sleep can take ~30–60 seconds while the container wakes up.
