"""Data validation for the training dataset using a Pandera schema."""
import pandas as pd
import pandera.pandas as pa
import pytest

REQUIRED = [
    "datetime", "Type", "Air_temperature", "Process_temperature",
    "Rotational_speed", "Torque", "Tool_wear", "Target", "RUL_hours", "Failure_Type",
]

# Schema reflects the same contract enforced by PreprocessingPipeline.validate_input.
SCHEMA = pa.DataFrameSchema(
    {
        "Type": pa.Column(str, pa.Check.isin(["H", "M", "L"])),
        "Air_temperature": pa.Column(float, pa.Check.in_range(200, 400)),
        "Process_temperature": pa.Column(float, pa.Check.in_range(200, 400)),
        "Rotational_speed": pa.Column(int, pa.Check.in_range(1000, 3000)),
        "Torque": pa.Column(float, pa.Check.in_range(0, 100)),
        "Tool_wear": pa.Column(int, pa.Check.in_range(0, 300)),
        "RUL_hours": pa.Column(int, pa.Check.ge(0)),
        "Target": pa.Column(int, pa.Check.isin([0, 1])),
        "month": pa.Column(int, pa.Check.in_range(1, 12)),
        "hour": pa.Column(int, pa.Check.in_range(0, 23)),
        "dayofweek": pa.Column(int, pa.Check.in_range(0, 6)),
    },
    strict=False,   # allow extra columns (machineID, Failure_Type, ...)
    coerce=True,
)


@pytest.fixture(scope="module")
def df(root):
    return pd.read_csv(root / "Data" / "predictive_maintenance.csv")


def test_required_columns_present(df):
    missing = set(REQUIRED) - set(df.columns)
    assert not missing, f"Missing columns: {missing}"


def test_no_missing_values(df):
    assert df[REQUIRED].isnull().sum().sum() == 0


def test_type_categories(df):
    assert set(df["Type"].unique()).issubset({"H", "M", "L"})


def test_schema_validates(df):
    # Raises SchemaErrors with a full report if any row violates the contract.
    SCHEMA.validate(df, lazy=True)
