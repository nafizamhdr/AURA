"""AURA Predictive Maintenance — Streamlit demo."""
import sys
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "Src"))
from Preprocessing_pipeline import PreprocessingPipeline  # noqa: E402

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "Model"
PIPELINE_PATH = ROOT / "Pipeline" / "preprocessing_pipeline.pkl"

st.set_page_config(
    page_title="AURA — Predictive Maintenance",
    page_icon="🔧",
    layout="wide",
)


@st.cache_resource
def load_models():
    pipeline = PreprocessingPipeline.load(str(PIPELINE_PATH))
    return {
        "pipeline": pipeline,
        "scaler": joblib.load(MODEL_DIR / "scaler.pkl"),
        "rul_model": joblib.load(MODEL_DIR / "rul_model.pkl"),
        "failure_model": joblib.load(MODEL_DIR / "failure_model.pkl"),
        "label_encoder": joblib.load(MODEL_DIR / "label_encoder.pkl"),
        "metadata": json.loads((MODEL_DIR / "model_metadata.json").read_text()),
    }


def predict(models, sensor_data):
    """Inference flow ported from Test.ipynb::predict_maintenance."""
    is_valid, error_msg = models["pipeline"].validate_input(sensor_data)
    if not is_valid:
        return {"status": "error", "error": error_msg}

    features = models["pipeline"].transform_single(sensor_data)
    features_scaled = models["scaler"].transform(features)

    rul_hours = float(models["rul_model"].predict(features_scaled)[0])
    rul_days = rul_hours / 24

    failure_type = None
    failure_confidence = None
    failure_probabilities = None
    if rul_days < 60:
        failure_idx = models["failure_model"].predict(features_scaled)[0]
        failure_proba = models["failure_model"].predict_proba(features_scaled)[0]
        failure_type = models["label_encoder"].inverse_transform([failure_idx])[0]
        failure_confidence = float(failure_proba[failure_idx])
        failure_probabilities = {
            cls: float(prob)
            for cls, prob in zip(models["label_encoder"].classes_, failure_proba)
        }

    if rul_days < 7:
        status, priority, action = "CRITICAL", "URGENT", "Schedule maintenance IMMEDIATELY (within 1–2 days)"
    elif rul_days < 30:
        status, priority, action = "CRITICAL", "HIGH", "Schedule maintenance within 1–2 weeks"
    elif rul_days < 60:
        status, priority, action = "WARNING", "MEDIUM", "Schedule maintenance within 4–8 weeks"
    else:
        status, priority, action = "NORMAL", "LOW", "Continue routine monitoring"

    return {
        "status": "success",
        "rul_hours": rul_hours,
        "rul_days": rul_days,
        "machine_status": status,
        "priority": priority,
        "action": action,
        "failure_type": failure_type,
        "failure_confidence": failure_confidence,
        "failure_probabilities": failure_probabilities,
    }


PRESETS = {
    "Scenario 1 — New machine, optimal": {
        "datetime": "2025-01-20 10:00:00", "Type": "M",
        "Air_temperature": 300.0, "Process_temperature": 310.0,
        "Rotational_speed": 1500, "Torque": 40.0, "Tool_wear": 50,
        "machine_age_hours": 1000, "hours_since_last": 8,
        "Temp_Rate_of_Change": 0.0, "RPM_Variance": 15.0,
    },
    "Scenario 2 — Aging, moderate wear": {
        "datetime": "2025-01-20 14:30:00", "Type": "H",
        "Air_temperature": 302.0, "Process_temperature": 310.0,
        "Rotational_speed": 1450, "Torque": 45.0, "Tool_wear": 180,
        "machine_age_hours": 15000, "hours_since_last": 8,
        "Temp_Rate_of_Change": 0.5, "RPM_Variance": 35.0,
    },
    "Scenario 3 — Critical, high wear": {
        "datetime": "2025-01-20 18:45:00", "Type": "L",
        "Air_temperature": 310.0, "Process_temperature": 322.0,
        "Rotational_speed": 1400, "Torque": 52.0, "Tool_wear": 280,
        "machine_age_hours": 22000, "hours_since_last": 8,
        "Temp_Rate_of_Change": 1.2, "RPM_Variance": 55.0,
    },
    "Scenario 4 — High RPM": {
        "datetime": "2025-01-20 22:15:00", "Type": "M",
        "Air_temperature": 298.0, "Process_temperature": 308.0,
        "Rotational_speed": 2800, "Torque": 35.0, "Tool_wear": 120,
        "machine_age_hours": 8000, "hours_since_last": 8,
        "Temp_Rate_of_Change": 0.2, "RPM_Variance": 45.0,
    },
    "Scenario 5 — Heavy load, high torque": {
        "datetime": "2025-01-21 08:00:00", "Type": "H",
        "Air_temperature": 302.0, "Process_temperature": 314.0,
        "Rotational_speed": 1350, "Torque": 68.0, "Tool_wear": 200,
        "machine_age_hours": 18000, "hours_since_last": 8,
        "Temp_Rate_of_Change": 0.8, "RPM_Variance": 40.0,
    },
}


def apply_preset(preset_name):
    data = PRESETS[preset_name]
    for k, v in data.items():
        st.session_state[f"in_{k}"] = v


def get_input(key, default):
    return st.session_state.get(f"in_{key}", default)


models = load_models()

st.title("🔧 AURA — Predictive Maintenance")
st.caption(
    f"Remaining Useful Life prediction using {models['metadata']['best_model']} "
    f"(Test MAE {models['metadata']['performance']['test_mae_days']:.1f} days, "
    f"R² {models['metadata']['performance']['test_r2']:.3f})."
)

with st.sidebar:
    st.header("Sensor Input")
    st.markdown("**Quick presets**")
    for preset_name in PRESETS:
        st.button(preset_name, on_click=apply_preset, args=(preset_name,), use_container_width=True)

    st.divider()
    st.markdown("**Machine**")
    machine_type = st.selectbox(
        "Type", ["H", "M", "L"],
        index=["H", "M", "L"].index(get_input("Type", "M")),
        key="in_Type",
    )

    st.markdown("**Sensor readings**")
    air_temp = st.slider("Air temperature (K)", 200.0, 400.0, float(get_input("Air_temperature", 300.0)), 0.1, key="in_Air_temperature")
    process_temp = st.slider("Process temperature (K)", 200.0, 400.0, float(get_input("Process_temperature", 310.0)), 0.1, key="in_Process_temperature")
    rpm = st.slider("Rotational speed (RPM)", 1000, 3000, int(get_input("Rotational_speed", 1500)), 10, key="in_Rotational_speed")
    torque = st.slider("Torque (Nm)", 0.0, 100.0, float(get_input("Torque", 40.0)), 0.5, key="in_Torque")
    tool_wear = st.slider("Tool wear (min)", 0, 300, int(get_input("Tool_wear", 100)), 1, key="in_Tool_wear")

    with st.expander("Optional / stateful features"):
        st.caption("Defaults are safe for single ad-hoc predictions. For streaming data, callers should track these per machine.")
        machine_age = st.number_input("Machine age (hours)", 0, 100000, int(get_input("machine_age_hours", 10000)), 100, key="in_machine_age_hours")
        hours_since = st.number_input("Hours since last reading", 0, 1000, int(get_input("hours_since_last", 8)), 1, key="in_hours_since_last")
        temp_roc = st.number_input("Temperature rate of change", -10.0, 10.0, float(get_input("Temp_Rate_of_Change", 0.0)), 0.1, key="in_Temp_Rate_of_Change")
        rpm_var = st.number_input("RPM variance", 0.0, 200.0, float(get_input("RPM_Variance", 20.0)), 1.0, key="in_RPM_Variance")

    predict_clicked = st.button("Predict", type="primary", use_container_width=True)

sensor_data = {
    "datetime": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Type": machine_type,
    "Air_temperature": air_temp,
    "Process_temperature": process_temp,
    "Rotational_speed": rpm,
    "Torque": torque,
    "Tool_wear": tool_wear,
    "machine_age_hours": machine_age,
    "hours_since_last": hours_since,
    "Temp_Rate_of_Change": temp_roc,
    "RPM_Variance": rpm_var,
}

if predict_clicked or "last_result" in st.session_state:
    if predict_clicked:
        st.session_state["last_result"] = predict(models, sensor_data)
        st.session_state["last_input"] = sensor_data
    result = st.session_state["last_result"]

    if result["status"] == "error":
        st.error(f"Validation failed: {result['error']}")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("RUL", f"{result['rul_days']:.1f} days", f"{result['rul_hours']:.0f} hours")
        col2.metric("Status", result["machine_status"])
        col3.metric("Priority", result["priority"])

        status = result["machine_status"]
        if status == "CRITICAL":
            st.error(f"🔴 **{status}** — {result['action']}")
        elif status == "WARNING":
            st.warning(f"🟡 **{status}** — {result['action']}")
        else:
            st.success(f"🟢 **{status}** — {result['action']}")

        rul_norm = min(result["rul_days"] / 365.0, 1.0)
        st.progress(rul_norm, text=f"RUL relative to 1 year: {rul_norm * 100:.0f}%")

        if result["failure_type"]:
            st.subheader("Failure type prediction")
            st.write(
                f"**Most likely:** {result['failure_type']}  ·  "
                f"**Confidence:** {result['failure_confidence']:.1%}"
            )
            proba_df = (
                pd.DataFrame(
                    {"Failure type": list(result["failure_probabilities"].keys()),
                     "Probability": list(result["failure_probabilities"].values())}
                )
                .sort_values("Probability", ascending=False)
                .set_index("Failure type")
            )
            st.bar_chart(proba_df)
        else:
            st.info("No imminent failure (RUL > 60 days). Failure classifier not invoked.")

        with st.expander("Show input payload"):
            st.json(st.session_state.get("last_input", sensor_data))
else:
    st.info("Configure sensor readings in the sidebar (or pick a preset) and click **Predict**.")

st.divider()
with st.expander("About this model"):
    st.markdown(
        f"""
        - **Task:** Predict Remaining Useful Life (hours) of industrial machines from sensor readings; classify likely failure type when RUL < 60 days.
        - **Best model:** {models['metadata']['best_model']} regressor, selected from {', '.join(models['metadata']['models_compared'])}.
        - **Training samples:** {models['metadata']['training_samples']:,} · **Test samples:** {models['metadata']['test_samples']:,}
        - **Features:** {models['metadata']['features']} (5 raw sensor + 5 engineered + 3 datetime + 2 temporal + 3 one-hot machine type)
        - **Failure classes:** {', '.join(models['label_encoder'].classes_)}
        """
    )
