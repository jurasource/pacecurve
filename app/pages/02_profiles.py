import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.cache import load_feature_matrix, load_profiler
from analysis.features import N_WINDOWS, WINDOW_COLS, WINDOW_MIDPOINT_HOURS
from analysis.profiles import PaceProfiler, select_n_clusters

st.title("Pace/Fatigue Profiles")

profiler = load_profiler()
feat, norm = load_feature_matrix()

col1, col2 = st.columns([3, 1])
with col2:
    n_clusters = st.slider("Number of profiles", 2, 5, 3)
    refit = st.button("Refit profiles")

# Refit if requested
if refit or n_clusters != profiler.n_clusters_:
    temp_profiler = PaceProfiler()
    temp_profiler.fit(norm, n_components=profiler.n_components_, n_clusters=n_clusters)
    display_profiler = temp_profiler
else:
    display_profiler = profiler

# Profile curves chart
st.subheader("Pace Profile Curves (normalised)")
fig1 = go.Figure()
colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
for k in range(display_profiler.n_clusters_):
    label = display_profiler.profile_labels_[k]
    curve = display_profiler.profile_curves_[k]
    fig1.add_trace(go.Scatter(
        x=WINDOW_MIDPOINT_HOURS,
        y=curve,
        mode="lines",
        name=label,
        line=dict(color=colors[k % len(colors)], width=2.5),
    ))
fig1.add_hline(y=1.0, line_dash="dot", line_color="grey", annotation_text="mean pace")
fig1.update_layout(
    xaxis_title="Race elapsed time (hours)",
    yaxis_title="Relative pace (1.0 = runner's mean)",
    height=400,
    legend=dict(orientation="h", y=-0.2),
)
st.plotly_chart(fig1)

st.caption(
    "Values below 1.0 = faster than the runner's own average; above 1.0 = slower. "
    "Profiles capture the *shape* of fatigue, not absolute pace."
)

# PCA scatter
st.subheader("Runner Distribution (PC1 vs PC2)")
X = norm[WINDOW_COLS].to_numpy(dtype=float)
valid = ~np.all(np.isnan(X), axis=1)
X_valid = X[valid]
col_means = np.nanmean(X_valid, axis=0)
nan_mask = np.isnan(X_valid)
X_valid[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

X_scaled = display_profiler.scaler_.transform(X_valid)
X_pca = display_profiler.pca_.transform(X_scaled)
labels_int = display_profiler.kmeans_.predict(X_pca)
profile_names = [display_profiler.profile_labels_[i] for i in labels_int]

meta_valid = norm[valid][["year", "name", "final_distance_km"]]
scatter_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "profile": profile_names,
    "year": meta_valid["year"].values,
    "name": meta_valid["name"].values,
    "final_km": meta_valid["final_distance_km"].values,
})

fig2 = go.Figure()
for k, label in enumerate(display_profiler.profile_labels_):
    sub = scatter_df[scatter_df["profile"] == label]
    fig2.add_trace(go.Scatter(
        x=sub["PC1"], y=sub["PC2"],
        mode="markers",
        name=label,
        marker=dict(color=colors[k % len(colors)], size=8, opacity=0.8),
        text=sub["year"].astype(str) + " — " + sub["name"] + f" ({sub['final_km'].round(1).astype(str)} km)",
        hovertemplate="%{text}<extra></extra>",
    ))
fig2.update_layout(
    xaxis_title="PC1",
    yaxis_title="PC2",
    height=450,
    legend=dict(orientation="h", y=-0.2),
)
st.plotly_chart(fig2)

# Variance explained
st.subheader("PCA Variance Explained")
var_explained = display_profiler.pca_.explained_variance_ratio_
fig3 = go.Figure(go.Bar(
    x=[f"PC{i+1}" for i in range(len(var_explained))],
    y=var_explained * 100,
    text=[f"{v*100:.1f}%" for v in var_explained],
    textposition="outside",
))
fig3.update_layout(
    yaxis_title="Variance explained (%)",
    height=300,
)
st.plotly_chart(fig3)

# Silhouette / elbow analysis
with st.expander("Cluster quality analysis (k=2..6)"):
    elbow = select_n_clusters(norm, max_k=6)
    col_a, col_b = st.columns(2)
    with col_a:
        fig4 = go.Figure(go.Scatter(x=elbow["k"], y=elbow["silhouette"], mode="lines+markers"))
        fig4.update_layout(xaxis_title="k", yaxis_title="Silhouette score", height=300)
        st.plotly_chart(fig4)
    with col_b:
        fig5 = go.Figure(go.Scatter(x=elbow["k"], y=elbow["inertia"], mode="lines+markers"))
        fig5.update_layout(xaxis_title="k", yaxis_title="Inertia (elbow method)", height=300)
        st.plotly_chart(fig5)

# Runners per profile table
st.subheader("Runners per Profile")
assignments = display_profiler.assign(norm)
norm_with_profile = norm.copy()
norm_with_profile["profile"] = assignments
table = (
    norm_with_profile[["year", "name", "final_distance_km", "gender", "profile"]]
    .sort_values(["profile", "final_distance_km"], ascending=[True, False])
    .rename(columns={"final_distance_km": "final_km", "gender": "Gender", "year": "Year", "name": "Name", "profile": "Profile"})
)
profile_filter = st.selectbox("Filter by profile", ["All"] + display_profiler.profile_labels_)
if profile_filter != "All":
    table = table[table["Profile"] == profile_filter]
st.dataframe(table, hide_index=True)
