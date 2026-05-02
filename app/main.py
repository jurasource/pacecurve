import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

st.set_page_config(
    page_title="PaceCurve",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = {
    "": [
        st.Page("pages/00_home.py", title="Overview", icon="🏠"),
    ],
    "Explore": [
        st.Page("pages/01_explore.py", title="Historical Data", icon="📊"),
        st.Page("pages/02_profiles.py", title="Pace Profiles", icon="📈"),
        st.Page("pages/03_backtest.py", title="Backtesting", icon="✅"),
        st.Page("pages/04_predict.py", title="Predict", icon="🎯"),
    ],
}

pg = st.navigation(pages)
pg.run()
