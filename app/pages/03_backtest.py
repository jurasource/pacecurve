import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.cache import load_backtest_metrics, load_backtest_results, load_per_profile_metrics

st.title("Backtesting Results")

st.markdown(
    "**Leave-One-Year-Out (LOYO):** For each of the 5 years, the model is trained on "
    "the other 4 years and evaluated on the held-out year. This gives an honest "
    "estimate of out-of-sample prediction accuracy."
)

bt = load_backtest_results()
metrics = load_backtest_metrics()

# MAE vs observation hours
st.subheader("Prediction Error vs Observation Time")
fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=metrics["obs_hours"], y=metrics["mae_km"],
    mode="lines+markers", name="MAE",
    line=dict(color="#e41a1c", width=2),
    marker=dict(size=8),
))
fig1.add_trace(go.Scatter(
    x=metrics["obs_hours"], y=metrics["rmse_km"],
    mode="lines+markers", name="RMSE",
    line=dict(color="#377eb8", width=2),
    marker=dict(size=8),
))
# CI band (p10/p90 of errors)
fig1.add_trace(go.Scatter(
    x=list(metrics["obs_hours"]) + list(metrics["obs_hours"])[::-1],
    y=list(metrics["p90_error_km"]) + list(metrics["p10_error_km"])[::-1],
    fill="toself",
    fillcolor="rgba(255,127,0,0.15)",
    line=dict(color="rgba(255,127,0,0)"),
    name="p10–p90 error range",
    showlegend=True,
))
fig1.add_hline(y=0, line_dash="dot", line_color="grey")
fig1.update_layout(
    xaxis_title="Observation time (hours)",
    yaxis_title="Error (km)",
    height=400,
)
st.plotly_chart(fig1)
st.caption("Error = predicted − actual. Negative = underestimate, positive = overestimate.")

col1, col2 = st.columns(2)
with col1:
    obs_h_disp = st.select_slider(
        "Observation horizon for scatter",
        options=sorted(bt["obs_hours"].unique()),
        value=6.0,
    )

# Predicted vs actual scatter
with col2:
    year_filter = st.multiselect("Year held out", sorted(bt["year_held_out"].unique()), default=sorted(bt["year_held_out"].unique()))

scatter_data = bt[(bt["obs_hours"] == obs_h_disp) & (bt["year_held_out"].isin(year_filter))]
fig2 = px.scatter(
    scatter_data,
    x="actual_km",
    y="predicted_km",
    color="year_held_out",
    hover_data=["name", "error_km"],
    labels={"actual_km": "Actual (km)", "predicted_km": "Predicted (km)", "year_held_out": "Year"},
    title=f"Predicted vs Actual at {obs_h_disp}h observation",
    color_continuous_scale=None,
)
max_val = max(scatter_data["actual_km"].max(), scatter_data["predicted_km"].max()) + 5
fig2.add_shape(type="line", x0=0, x1=max_val, y0=0, y1=max_val, line=dict(dash="dot", color="grey"))
fig2.update_layout(height=450)
st.plotly_chart(fig2)

# Metrics table
st.subheader("Summary Metrics by Observation Hour")
display_metrics = metrics[["obs_hours", "n", "mae_km", "rmse_km", "median_error_km", "p10_error_km", "p90_error_km"]].copy()
display_metrics.columns = ["Hours", "N", "MAE (km)", "RMSE (km)", "Median error", "p10 error", "p90 error"]
display_metrics = display_metrics.round(1)
st.dataframe(display_metrics, hide_index=True)

# Per-runner table
st.subheader("Per-Runner Predictions")
obs_h_table = st.select_slider(
    "Observation horizon for table",
    options=sorted(bt["obs_hours"].unique()),
    value=6.0,
    key="table_slider",
)
table_data = (
    bt[bt["obs_hours"] == obs_h_table]
    .sort_values("abs_error_km" if "abs_error_km" in bt.columns else "error_km", ascending=True)
    [["year_held_out", "name", "actual_km", "predicted_km", "error_km", "profile_label"]]
    .rename(columns={
        "year_held_out": "Year",
        "name": "Name",
        "actual_km": "Actual (km)",
        "predicted_km": "Predicted (km)",
        "error_km": "Error (km)",
        "profile_label": "Profile",
    })
    .round(1)
)
st.dataframe(table_data, hide_index=True)
