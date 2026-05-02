import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app.cache import load_backtest_metrics, load_backtest_results, load_feature_matrix, load_profiler
from analysis.features import WINDOW_MIDPOINT_HOURS

st.title("PaceCurve")
st.caption("Real-time distance prediction for 24-hour ultra running races")

st.divider()

# --- Dataset snapshot ---
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Years of data", "5", "2021 – 2025")
col_b.metric("Runners", "235", "181 finishers")
col_c.metric("Lap records", "89,707")
col_d.metric("Race", "Self-Transcendence 24hr", "London")

st.divider()

# --- How it works ---
st.subheader("How it works")

step_col1, step_col2, step_col3 = st.columns(3)

with step_col1:
    st.markdown("**1. Feature engineering**")
    st.markdown(
        "Each runner's lap sequence is divided into 48 consecutive 30-minute windows. "
        "The mean lap time within each window becomes one feature, giving a 48-dimensional "
        "pace trajectory. Each trajectory is then normalised by the runner's own mean pace "
        "so that the model captures *shape* — how pace changes relative to the runner's "
        "average — rather than absolute speed."
    )

with step_col2:
    st.markdown("**2. Profile extraction**")
    st.markdown(
        "PCA reduces the 48-dimensional trajectories to 10 principal components, then "
        "k-means clustering identifies 3 archetypal pace/fatigue profiles. The cluster "
        "centroids are inverted back to pace-curve space to give interpretable profile "
        "shapes labelled **Aggressive**, **Conservative**, and **Steady** based on the "
        "slope of each curve's back half."
    )

with step_col3:
    st.markdown("**3. Real-time prediction**")
    st.markdown(
        "As a race unfolds, observed lap times are fitted to each profile via least-squares "
        "scale fitting. The best-matching profile is used to extrapolate the runner's future "
        "pace across the remaining windows, and total laps are simulated second-by-second "
        "to produce a distance prediction. With fewer than 3 hours of data, a simpler "
        "constant-pace fallback is used."
    )

st.divider()

# --- Summary charts ---
st.subheader("Model performance")
st.caption(
    "Leave-One-Year-Out cross-validation: trained on 4 years, tested on the held-out year. "
    "Repeated for all 5 years."
)

metrics = load_backtest_metrics()
profiler = load_profiler()
_, norm = load_feature_matrix()

chart_col1, chart_col2 = st.columns(2)

# Chart 1: MAE vs observation time
with chart_col1:
    fig_mae = go.Figure()
    fig_mae.add_trace(go.Scatter(
        x=metrics["obs_hours"],
        y=metrics["mae_km"],
        mode="lines+markers",
        name="MAE",
        line=dict(color="#e41a1c", width=2.5),
        marker=dict(size=8),
        fill="tozeroy",
        fillcolor="rgba(228,26,28,0.08)",
    ))
    fig_mae.add_trace(go.Scatter(
        x=metrics["obs_hours"],
        y=metrics["rmse_km"],
        mode="lines+markers",
        name="RMSE",
        line=dict(color="#377eb8", width=2, dash="dot"),
        marker=dict(size=6),
    ))
    fig_mae.update_layout(
        title="Prediction error vs. observation time",
        xaxis_title="Hours of race observed",
        yaxis_title="Error (km)",
        height=320,
        legend=dict(orientation="h", y=-0.25),
        margin=dict(t=40, b=10),
    )
    st.plotly_chart(fig_mae)
    st.caption(
        "Error falls from ~70 km at 1h (constant-pace tier) to ~7 km at 18h. "
        "The big drop between 1h and 3h reflects the switch to profile-based prediction."
    )

# Chart 2: The 3 profile curves
with chart_col2:
    colors = ["#e41a1c", "#377eb8", "#4daf4a"]
    fig_profiles = go.Figure()
    for k in range(profiler.n_clusters_):
        fig_profiles.add_trace(go.Scatter(
            x=WINDOW_MIDPOINT_HOURS,
            y=profiler.profile_curves_[k],
            mode="lines",
            name=profiler.profile_labels_[k],
            line=dict(color=colors[k], width=2.5),
        ))
    fig_profiles.add_hline(
        y=1.0, line_dash="dot", line_color="grey",
        annotation_text="mean pace", annotation_position="bottom right",
    )
    fig_profiles.update_layout(
        title="The 3 pace/fatigue archetypes",
        xaxis_title="Race elapsed time (hours)",
        yaxis_title="Relative pace (1.0 = runner's mean)",
        height=320,
        legend=dict(orientation="h", y=-0.25),
        margin=dict(t=40, b=10),
    )
    st.plotly_chart(fig_profiles)
    st.caption(
        "Below 1.0 = faster than the runner's own average; above 1.0 = slower. "
        "All three profiles converge on fatigue, but at different rates."
    )

# Key numbers row
st.divider()
m6 = metrics[metrics["obs_hours"] == 6].iloc[0]
m12 = metrics[metrics["obs_hours"] == 12].iloc[0]
m18 = metrics[metrics["obs_hours"] == 18].iloc[0]

res_col1, res_col2, res_col3, res_col4 = st.columns(4)
res_col1.metric("MAE at 6h", f"{m6['mae_km']:.1f} km", help="Mean absolute error with 6 hours of observed data")
res_col2.metric("MAE at 12h", f"{m12['mae_km']:.1f} km", help="Mean absolute error with 12 hours of observed data")
res_col3.metric("MAE at 18h", f"{m18['mae_km']:.1f} km", help="Mean absolute error with 18 hours of observed data")
res_col4.metric("Profiles", str(profiler.n_clusters_), ", ".join(profiler.profile_labels_))

st.divider()

# --- Limitations & improvement areas ---
st.subheader("Known limitations & areas for improvement")

lim_col1, lim_col2 = st.columns(2)

with lim_col1:
    st.markdown("**Current limitations**")
    st.markdown(
        "- **Early-race predictions are poor.** With fewer than ~3 hours of data the model "
        "falls back to constant-pace extrapolation, which systematically overestimates "
        "because runners slow down. MAE at 1h is ~70 km.\n"
        "- **Small training set.** Only 181 finishers across 5 years means the 3 profiles "
        "have soft boundaries — silhouette scores are modest (~0.14 at k=3). A larger "
        "dataset would sharpen the clusters.\n"
        "- **Single race, single course.** All data comes from one event in London. The "
        "profiles may not generalise to other 24hr races with different conditions or "
        "surfaces.\n"
        "- **No runner fitness baseline.** Two runners with very different training "
        "backgrounds but similar early paces receive identical predictions. A fitness "
        "factor (e.g. from Strava) would help differentiate them.\n"
        "- **DNF runners ignored in training.** Runners who dropped out are excluded from "
        "profile fitting, so the model can't predict likelihood of DNF."
    )

with lim_col2:
    st.markdown("**Priority improvements**")
    st.markdown(
        "- **Weather integration.** Race-day temperature and humidity are correlated with "
        "late-race slowdown. Adding historical weather data (date + location → API) could "
        "meaningfully shift profile curve shapes.\n"
        "- **Strava fitness factor.** A runner's recent training load provides a prior on "
        "their expected finishing distance, independent of early race pace.\n"
        "- **More race data.** Scraping other 24hr races (e.g. Sri Chinmoy, IAU World "
        "Championships) would increase training data and produce more robust profiles.\n"
        "- **Probabilistic output.** Replace the heuristic ±15% confidence interval with "
        "a proper uncertainty estimate derived from the profile residuals.\n"
        "- **Fixed-distance support.** The feature engineering and profile approach would "
        "extend naturally to predicting finish time in fixed-distance races (100km, 100mi), "
        "with elevation profile as an additional input."
    )

st.divider()
st.caption(
    "Data sourced from [raceresult.com](https://raceresult.com) via the Self-Transcendence "
    "24 Hour Track Race results pages."
)
