"""Shared cached resources for all Streamlit pages."""
from pathlib import Path

import pandas as pd
import streamlit as st

MODELS_DIR = Path(__file__).parent.parent / "models"


@st.cache_resource
def load_profiler():
    from analysis.profiles import PaceProfiler
    return PaceProfiler.load(MODELS_DIR / "profiler.pkl")


@st.cache_resource
def get_db():
    from analysis.data import get_connection
    return get_connection()


@st.cache_data
def load_all_laps():
    from analysis.data import get_all_laps
    conn = get_db()
    return get_all_laps(conn, exclude_dnf=False)


@st.cache_data
def load_feature_matrix():
    from analysis.data import get_all_laps
    from analysis.features import build_feature_matrix, normalise_features
    conn = get_db()
    laps_no_dnf = get_all_laps(conn, exclude_dnf=True)
    feat = build_feature_matrix(laps_no_dnf)
    norm = normalise_features(feat)
    return feat, norm


@st.cache_data
def load_backtest_results():
    return pd.read_parquet(MODELS_DIR / "backtest_results.parquet")


@st.cache_data
def load_backtest_metrics():
    return pd.read_parquet(MODELS_DIR / "backtest_metrics.parquet")


@st.cache_data
def load_per_profile_metrics():
    return pd.read_parquet(MODELS_DIR / "backtest_per_profile.parquet")
