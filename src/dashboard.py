from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

try:
    from .data_loader import filter_discharge_cycles, load_nasa_battery_directory
    from .features import DEFAULT_EOL_SOH_THRESHOLD, build_training_frame
    from .models import anomaly_to_risk_level
except ImportError:
    from data_loader import filter_discharge_cycles, load_nasa_battery_directory
    from features import DEFAULT_EOL_SOH_THRESHOLD, build_training_frame
    from models import anomaly_to_risk_level


st.set_page_config(page_title="EV Battery Health V1", layout="wide")
st.title("EV Battery Health Analyzer (V1)")


@st.cache_data
def load_frame(data_dir: str, eol_soh_threshold: float) -> pd.DataFrame:
    raw_df = load_nasa_battery_directory(data_dir)
    return build_training_frame(
        filter_discharge_cycles(raw_df), eol_soh_threshold=eol_soh_threshold
    )


@st.cache_resource
def load_bundle(path: str):
    if Path(path).exists():
        return joblib.load(path)
    return None


data_dir = st.sidebar.text_input("Data directory", "data")
artifact_path = st.sidebar.text_input("Model artifact", "artifacts/battery_model_bundle.joblib")
eol_soh_threshold = st.sidebar.slider(
    "EOL SOH threshold", min_value=0.6, max_value=0.95, value=float(DEFAULT_EOL_SOH_THRESHOLD), step=0.01
)
frame = load_frame(data_dir, eol_soh_threshold)
bundle = load_bundle(artifact_path)

selected_battery = st.sidebar.selectbox("Battery", sorted(frame["battery_id"].unique().tolist()))
battery_df = frame.loc[frame["battery_id"] == selected_battery].copy()

if bundle is not None:
    x = battery_df[bundle.feature_columns].copy()
    battery_df["soh_pred"] = bundle.soh_model.predict(x)
    battery_df["rul_pred_cycles"] = bundle.rul_model.predict(x).clip(min=0.0)
    battery_df["anomaly_score"] = bundle.anomaly_model.decision_function(x)
    battery_df["risk_level"] = anomaly_to_risk_level(battery_df["anomaly_score"].values)
else:
    st.warning("No trained model artifact found. Train first with `python -m src.train`.")
    battery_df["soh_pred"] = battery_df["soh"]
    battery_df["rul_pred_cycles"] = battery_df["rul_cycles"]
    battery_df["anomaly_score"] = 0.0
    battery_df["risk_level"] = "medium"

col1, col2, col3 = st.columns(3)
col1.metric("Latest SOH (pred)", f"{battery_df['soh_pred'].iloc[-1] * 100:.2f}%")
col2.metric("Latest RUL (pred)", f"{battery_df['rul_pred_cycles'].iloc[-1]:.1f} cycles")
col3.metric(
    "High-risk cycles",
    int((battery_df["risk_level"] == "high").sum()),
)

st.subheader(f"Capacity Fade: {selected_battery}")
st.line_chart(
    battery_df.set_index("cycle_number")[["capacity_ah", "capacity_roll5_mean_ah"]],
    use_container_width=True,
)

st.subheader("SOH Actual vs Predicted")
st.caption(f"EOL threshold: SOH <= {eol_soh_threshold:.2f}")
st.line_chart(
    battery_df.set_index("cycle_number")[["soh", "soh_pred"]],
    use_container_width=True,
)

st.subheader("RUL Actual vs Predicted (EOL-based)")
st.line_chart(
    battery_df.set_index("cycle_number")[["rul_cycles", "rul_pred_cycles"]],
    use_container_width=True,
)

st.subheader("Risk Distribution")
st.bar_chart(battery_df["risk_level"].value_counts())

st.subheader("Recent Cycles")
st.dataframe(
    battery_df[
        [
            "cycle_number",
            "capacity_ah",
            "soh",
            "soh_pred",
            "rul_pred_cycles",
            "anomaly_score",
            "risk_level",
        ]
    ].tail(20),
    use_container_width=True,
)
