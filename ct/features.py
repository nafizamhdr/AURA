"""Deterministic feature engineering for AURA (replicates Src/FE.ipynb).

Recomputes the full 18-feature matrix from the raw dataset using the machine
time-series (machineID + datetime), so training is reproducible from the
committed raw CSV alone -- no separate engineered data file is required.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURE_ORDER = json.loads((ROOT / "Model" / "features.json").read_text())


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the engineered/temporal/one-hot columns (matches FE.ipynb)."""
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["machineID", "datetime"]).reset_index(drop=True)

    # Temporal features (per machine)
    df["machine_age_hours"] = df.groupby("machineID").cumcount() * 8
    df["hours_since_last"] = (
        df.groupby("machineID")["datetime"].diff().dt.total_seconds() / 3600
    ).fillna(8)

    # Engineered sensor features
    df["Temp_Difference"] = df["Process_temperature"] - df["Air_temperature"]
    df["Temp_Rate_of_Change"] = df.groupby("machineID")["Process_temperature"].diff().fillna(0)
    df["Power"] = df["Torque"] * df["Rotational_speed"] / 9.5488
    df["Torque_Speed_Ratio"] = df["Torque"] / (df["Rotational_speed"] + 1)
    df["RPM_Variance"] = (
        df.groupby("machineID")["Rotational_speed"]
        .transform(lambda x: x.rolling(window=min(50, len(x)), min_periods=1).std())
        .fillna(0)
    )

    # Datetime parts (recomputed for safety)
    df["month"] = df["datetime"].dt.month
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek

    # One-hot machine type
    df["Type_H"] = (df["Type"] == "H").astype(int)
    df["Type_L"] = (df["Type"] == "L").astype(int)
    df["Type_M"] = (df["Type"] == "M").astype(int)
    return df


def features_and_target(df: pd.DataFrame):
    """Return (X[n,18], y, engineered_df) ready for training."""
    eng = engineer_features(df)
    X = eng[FEATURE_ORDER].to_numpy(dtype=float)
    y = eng["RUL_hours"].to_numpy(dtype=float)
    return X, y, eng
