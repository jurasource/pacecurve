import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.cache import load_all_laps, load_feature_matrix, load_profiler
from analysis.features import DNF_STATUS, N_WINDOWS, WINDOW_MIDPOINT_HOURS

st.title("Historical Data Explorer")

laps = load_all_laps()
profiler = load_profiler()
_, norm = load_feature_matrix()

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    all_years = sorted(laps["year"].unique())
    selected_years = st.multiselect("Years", all_years, default=all_years)
    selected_genders = st.multiselect("Gender", ["M", "F"], default=["M", "F"])
    show_dnf = st.toggle("Include DNF runners", value=False)

if not selected_years:
    st.warning("Select at least one year.")
    st.stop()

filtered_laps = laps[laps["year"].isin(selected_years)]
if selected_genders:
    filtered_laps = filtered_laps[filtered_laps["gender"].isin(selected_genders)]
if not show_dnf:
    filtered_laps = filtered_laps[filtered_laps["status"].isna() | (filtered_laps["status"] != DNF_STATUS)]

# Summary stats — computed from laps already in memory, no extra DB call
st.subheader("Summary")
summary_rows = []
for year in selected_years:
    year_runners = (
        filtered_laps[filtered_laps["year"] == year]
        .groupby("pid")["final_distance_km"]
        .first()
        .dropna()
    )
    summary_rows.append({
        "Year": year,
        "Runners": len(year_runners),
        "Avg distance (km)": f"{year_runners.mean():.1f}" if len(year_runners) else "-",
        "Winner (km)": f"{year_runners.max():.1f}" if len(year_runners) else "-",
        "Min (km)": f"{year_runners.min():.1f}" if len(year_runners) else "-",
    })
st.dataframe(pd.DataFrame(summary_rows), hide_index=True)

# Pace trajectory chart
st.subheader("Pace Trajectory")

runner_options = (
    filtered_laps.groupby(["year", "name"])["final_distance_km"]
    .first()
    .reset_index()
    .assign(label=lambda df: df["year"].astype(str) + " — " + df["name"] + f" ({df['final_distance_km'].round(1).astype(str)} km)")
    .sort_values(["year", "final_distance_km"], ascending=[True, False])
)

col1, col2 = st.columns([2, 1])
with col1:
    selected_runners = st.multiselect(
        "Select runners to plot",
        options=runner_options["label"].tolist(),
        default=runner_options["label"].tolist()[:3],
    )
with col2:
    smoothing = st.slider("Rolling average (laps)", 1, 20, 5)
    show_profile = st.toggle("Overlay profile curve", value=False)

fig = go.Figure()

for label in selected_runners:
    row = runner_options[runner_options["label"] == label].iloc[0]
    year, name = int(row["year"]), row["name"]
    runner_laps = filtered_laps[(filtered_laps["year"] == year) & (filtered_laps["name"] == name)].sort_values("lap_number")
    if runner_laps.empty:
        continue

    elapsed_hours = runner_laps["split_time_sec"].cumsum() / 3600
    pace = runner_laps["split_time_sec"].rolling(smoothing, min_periods=1).mean()

    fig.add_trace(go.Scatter(
        x=elapsed_hours,
        y=pace,
        mode="lines",
        name=label,
        line=dict(width=1.5),
    ))

if show_profile and selected_runners:
    row = runner_options[runner_options["label"] == selected_runners[0]].iloc[0]
    yr, nm = int(row["year"]), row["name"]
    match = filtered_laps[(filtered_laps["year"] == yr) & (filtered_laps["name"] == nm)]
    if not match.empty:
        eid = match["event_id"].iloc[0]
        pid = match["pid"].iloc[0]
        key = (eid, pid)
        if key in norm.index:
            # Assign only this one runner, not the entire 181-row matrix
            profile_name = profiler.assign(norm.loc[[key]]).iloc[0]
            profile_idx = profiler.profile_labels_.index(profile_name)
            curve = profiler.profile_curves_[profile_idx]
            runner_mean_pace = match["split_time_sec"].mean()
            fig.add_trace(go.Scatter(
                x=WINDOW_MIDPOINT_HOURS,
                y=curve * runner_mean_pace,
                mode="lines",
                name=f"{profile_name} profile",
                line=dict(dash="dash", width=2, color="black"),
            ))

fig.update_layout(
    xaxis_title="Elapsed time (hours)",
    yaxis_title="Lap time (sec/lap)",
    yaxis=dict(autorange="reversed"),
    legend=dict(orientation="h", y=-0.2),
    height=500,
)
st.plotly_chart(fig)

# Distance scatter
st.subheader("Early Pace vs Final Distance")
_sorted = filtered_laps.sort_values(["year", "name", "lap_number"])
_early = (
    _sorted.groupby(["year", "name"])["split_time_sec"]
    .apply(lambda s: s.iloc[:20].mean())  # first 20 laps ≈ first ~8 minutes
    .rename("early_pace")
    .reset_index()
)
_meta = (
    _sorted.groupby(["year", "name"])[["gender", "final_distance_km"]]
    .first()
    .reset_index()
)
runners_plot = _meta.merge(_early, on=["year", "name"])

fig2 = go.Figure()
for gender in ["M", "F"]:
    subset = runners_plot[runners_plot["gender"] == gender]
    fig2.add_trace(go.Scatter(
        x=subset["early_pace"],
        y=subset["final_distance_km"],
        mode="markers",
        name=gender,
        text=subset["year"].astype(str) + " — " + subset["name"],
        marker=dict(size=8),
    ))
fig2.update_layout(
    xaxis_title="Mean pace of first 20 laps (sec/lap)",
    yaxis_title="Final distance (km)",
    height=400,
)
st.plotly_chart(fig2)
