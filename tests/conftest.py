"""Shared pytest fixtures and path setup for AURA CI tests."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Src"))

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
