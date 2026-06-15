"""Request/response models for the AURA prediction API."""
from typing import Optional

from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    """One sensor reading. Required fields mirror PreprocessingPipeline; the
    optional stateful features are used when provided and defaulted otherwise."""
    Type: str = Field(..., description="Machine type: H, M, or L")
    Air_temperature: float = Field(..., description="Kelvin, 200-400")
    Process_temperature: float = Field(..., description="Kelvin, 200-400")
    Rotational_speed: float = Field(..., description="RPM, 1000-3000")
    Torque: float = Field(..., description="Nm, 0-100")
    Tool_wear: float = Field(..., description="minutes, 0-300")

    datetime: Optional[str] = None
    machine_age_hours: Optional[float] = None
    hours_since_last: Optional[float] = None
    Temp_Rate_of_Change: Optional[float] = None
    RPM_Variance: Optional[float] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "Type": "M", "Air_temperature": 300.0, "Process_temperature": 310.0,
                "Rotational_speed": 1500, "Torque": 40.0, "Tool_wear": 100,
            }
        }
    }


class PredictionResponse(BaseModel):
    rul_hours: float
    rul_days: float
    machine_status: str
    priority: str
    action: str
    failure_type: Optional[str] = None
    failure_confidence: Optional[float] = None
    failure_probabilities: Optional[dict] = None
