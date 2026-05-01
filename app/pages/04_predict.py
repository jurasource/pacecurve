import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.cache import load_all_laps, load_profiler
from analysis.features import WINDOW_SEC
from analysis.predictor import Predictor

st.title("Real-Time Prediction")

profiler = load_profiler()
predictor = Predictor(profiler)

# Input mode
input_mode = st.radio(
    "Input method",
    ["Historical runner (demo)", "Manual lap entry", "CSV upload"],
    horizontal=True,
)

laps_observed: list[dict] = []
actual_km: float | None = None
runner_label = ""

if input_mode == "Historical runner (demo)":
    all_laps = load_all_laps()
    non_dnf = all_laps[all_laps["status"].isna() | (all_laps["status"] != "*")]
    runner_options = (
        non_dnf.groupby(["year", "name"])["final_distance_km"]
        .first()
        .reset_index()
        .sort_values(["year", "final_distance_km"], ascending=[True, False])
        .assign(label=lambda df: df["year"].astype(str) + " — " + df["name"] + f" ({df['final_distance_km'].round(1).astype(str)} km)")
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        selected = st.selectbox("Select runner", runner_options["label"].tolist())
    with col2:
        obs_hours = st.slider("Observation time (hours)", 1, 23, 6)

    row = runner_options[runner_options["label"] == selected].iloc[0]
    year, name = int(row["year"]), row["name"]
    actual_km = float(row["final_distance_km"])
    runner_label = selected

    runner_laps = non_dnf[(non_dnf["year"] == year) & (non_dnf["name"] == name)].sort_values("lap_number")
    elapsed_cumsum = runner_laps["split_time_sec"].cumsum()
    partial = runner_laps[elapsed_cumsum <= obs_hours * 3600]
    laps_observed = [
        {"lap_number": int(r["lap_number"]), "split_time_sec": int(r["split_time_sec"])}
        for _, r in partial.iterrows()
    ]

elif input_mode == "Manual lap entry":
    st.markdown("Enter lap split times in seconds (one per row or comma-separated):")
    raw_input = st.text_area("Split times (sec/lap)", placeholder="133\n134\n138\n...", height=200)
    if raw_input.strip():
        try:
            values = [float(v.strip()) for v in raw_input.replace(",", "\n").split("\n") if v.strip()]
            laps_observed = [{"lap_number": i + 1, "split_time_sec": v} for i, v in enumerate(values)]
            runner_label = f"{len(values)} laps entered manually"
        except ValueError:
            st.error("Could not parse input. Enter one number per line.")

elif input_mode == "CSV upload":
    uploaded = st.file_uploader("CSV with columns: lap_number, split_time_sec", type="csv")
    if uploaded:
        df_upload = pd.read_csv(uploaded)
        if "split_time_sec" not in df_upload.columns:
            st.error("CSV must have a 'split_time_sec' column.")
        else:
            if "lap_number" not in df_upload.columns:
                df_upload["lap_number"] = range(1, len(df_upload) + 1)
            laps_observed = df_upload[["lap_number", "split_time_sec"]].to_dict("records")
            runner_label = uploaded.name

# --- Output ---
if not laps_observed:
    st.info("Enter or select lap data above to generate a prediction.")
    st.stop()

result = predictor.predict(laps_observed)
trajectory = predictor.predict_trajectory(laps_observed)

st.divider()

# Key metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Predicted distance", f"{result['predicted_km']} km")
with col2:
    st.metric("In miles", f"{result['predicted_miles']} mi")
with col3:
    ci = result["confidence_interval_km"]
    st.metric("Confidence interval", f"{ci[0]}–{ci[1]} km")
with col4:
    st.metric("Pace profile", result["profile_label"])

if actual_km is not None:
    st.success(f"Actual result: **{actual_km:.1f} km** — Prediction error: **{result['predicted_km'] - actual_km:+.1f} km**")

if result["tier"] == 1:
    st.warning(
        f"Only {result['hours_observed']:.1f}h of data observed — using constant-pace extrapolation (Tier 1). "
        "Accuracy improves significantly after 3+ hours of data."
    )

# Race progress bar
obs_frac = min(result["hours_observed"] / 24.0, 1.0)
st.progress(obs_frac, text=f"Race progress: {result['hours_observed']:.1f}h / 24h observed")

st.subheader("Pace Trajectory")
hours = trajectory["hours"]
obs_pace = trajectory["observed_pace"]
pred_pace = trajectory["predicted_pace"]

obs_hours_list = [h for h, v in zip(hours, obs_pace) if not np.isnan(v)]
obs_vals = [v for v in obs_pace if not np.isnan(v)]
pred_hours_future = [h for h, v in zip(hours, obs_pace) if np.isnan(v)]
pred_vals_future = [pred_pace[i] for i, v in enumerate(obs_pace) if np.isnan(v)]

fig = go.Figure()

# Observed pace
fig.add_trace(go.Scatter(
    x=obs_hours_list,
    y=obs_vals,
    mode="lines",
    name="Observed",
    line=dict(color="#1f77b4", width=2),
))

# Predicted future pace
if pred_hours_future:
    if obs_hours_list:
        # Connect observed to predicted
        transition_x = [obs_hours_list[-1]] + pred_hours_future
        transition_y = [obs_vals[-1]] + pred_vals_future
    else:
        transition_x = pred_hours_future
        transition_y = pred_vals_future

    fig.add_trace(go.Scatter(
        x=transition_x,
        y=transition_y,
        mode="lines",
        name="Predicted",
        line=dict(color="#ff7f0e", width=2, dash="dash"),
    ))

# Mark prediction point
if actual_km is not None:
    fig.add_vline(x=result["hours_observed"], line_dash="dot", line_color="grey",
                  annotation_text=f"{result['hours_observed']:.1f}h cutoff")

fig.update_layout(
    xaxis_title="Race elapsed time (hours)",
    yaxis_title="Lap time (sec/lap)",
    yaxis=dict(autorange="reversed"),
    height=400,
    legend=dict(orientation="h", y=-0.2),
)
st.plotly_chart(fig)

# Detailed prediction table
with st.expander("Prediction details"):
    details = {
        "Input": runner_label,
        "Laps observed": len(laps_observed),
        "Hours observed": f"{result['hours_observed']:.2f}h",
        "Predicted km": result["predicted_km"],
        "Predicted laps": result["predicted_laps"],
        "Confidence interval": f"{ci[0]}–{ci[1]} km",
        "Profile": result["profile_label"],
        "Prediction tier": f"Tier {result['tier']} ({'constant pace' if result['tier']==1 else 'profile-based'})",
    }
    if actual_km is not None:
        details["Actual km"] = actual_km
        details["Error (km)"] = f"{result['predicted_km'] - actual_km:+.1f}"
    st.table(pd.DataFrame({"Value": {k: str(v) for k, v in details.items()}}))
