import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.cache import get_db, load_all_laps, load_feature_matrix, load_profiler
from analysis.data import get_participants

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
    filtered_laps = filtered_laps[filtered_laps["status"].isna() | (filtered_laps["status"] != "*")]

# Summary stats
st.subheader("Summary")
conn = get_db()
participants = get_participants(conn)
participants["distance_km"] = pd.to_numeric(participants["distance_km"], errors="coerce")

summary_rows = []
for year in selected_years:
    year_laps = filtered_laps[filtered_laps["year"] == year]
    year_parts = participants[participants["event_id"].isin(year_laps["event_id"].unique())]
    year_parts_filtered = year_parts[year_parts["gender"].isin(selected_genders)] if selected_genders else year_parts
    if not show_dnf:
        year_parts_filtered = year_parts_filtered[year_parts_filtered["status"].isna() | (year_parts_filtered["status"] != "*")]
    dist = pd.to_numeric(year_parts_filtered["distance_km"], errors="coerce").dropna()
    summary_rows.append({
        "Year": year,
        "Runners": len(dist),
        "Avg distance (km)": f"{dist.mean():.1f}" if len(dist) else "-",
        "Winner (km)": f"{dist.max():.1f}" if len(dist) else "-",
        "Min (km)": f"{dist.min():.1f}" if len(dist) else "-",
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

    elapsed_sec = runner_laps["split_time_sec"].cumsum()
    elapsed_hours = elapsed_sec / 3600
    pace = runner_laps["split_time_sec"].rolling(smoothing, min_periods=1).mean()

    fig.add_trace(go.Scatter(
        x=elapsed_hours,
        y=pace,
        mode="lines",
        name=label,
        line=dict(width=1.5),
    ))

if show_profile and selected_runners:
    assignments = profiler.assign(norm)
    for label in selected_runners[:1]:
        row = runner_options[runner_options["label"] == label].iloc[0]
        idx = (int(row["event_id"]) if "event_id" in row else None, row["pid"] if "pid" in row else None)
        # Use year + name to find event_id/pid
        yr, nm = int(row["year"]), row["name"]
        match = filtered_laps[(filtered_laps["year"] == yr) & (filtered_laps["name"] == nm)]
        if not match.empty:
            eid = match["event_id"].iloc[0]
            pid = match["pid"].iloc[0]
            key = (eid, pid)
            if key in assignments.index:
                profile_name = assignments[key]
                profile_idx = profiler.profile_labels_.index(profile_name)
                curve = profiler.profile_curves_[profile_idx]
                runner_mean_pace = match["split_time_sec"].mean()
                abs_curve = curve * runner_mean_pace
                hours = [(i + 0.5) * 0.5 for i in range(48)]
                fig.add_trace(go.Scatter(
                    x=hours, y=abs_curve,
                    mode="lines", name=f"{profile_name} profile",
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
    .apply(lambda s: s.iloc[:20].mean())
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
