# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Launch the Streamlit app
streamlit run app/main.py

# Train/retrain model artefacts (must be run before the app will load)
python scripts/train.py

# Re-scrape race data from raceresult.com
python scraper.py
```

The app will fail to start if `models/profiler.pkl`, `models/backtest_results.parquet`, `models/backtest_metrics.parquet`, and `models/backtest_per_profile.parquet` don't exist — run `scripts/train.py` first.

## Architecture

Two independent layers share `race_data.db` (SQLite, ~15 MB):

1. **Scraper layer** (root: `scraper.py`, `raceresult.py`, `eventrac.py`, `config.py`, `db.py`) — fetches Self-Transcendence 24hr Track Race results from raceresult.com. Standalone; the analysis layer never imports from it.

2. **Analysis + App layer** (`analysis/`, `app/`, `scripts/`) — all analytical code plus the Streamlit UI.

Data flow through the analysis layer:

```
race_data.db
  → analysis/data.py          (DB queries → DataFrames)
  → analysis/features.py      (raw laps → 48-window feature matrix)
  → analysis/profiles.py      (PCA + k-means → PaceProfiler)
  → analysis/predictor.py     (partial laps → distance prediction)
  → analysis/backtest.py      (LOYO evaluation)
  → scripts/train.py          (fits everything, writes models/)
  → app/cache.py              (Streamlit @cache_resource / @cache_data wrappers)
  → app/pages/                (UI pages)
```

## Critical Data Quirk

`elapsed_time_sec` in the `laps` table is NULL for ~94% of rows (scraper bug with `H:MM:SS.ss` parsing). **Always reconstruct elapsed time as `split_time_sec.cumsum()` sorted by `lap_number`** within each (event_id, pid) group. Never read the stored `elapsed_time_sec` column.

## Key Constants (all in `analysis/features.py`)

- `N_WINDOWS = 48` — 30-minute windows across the 24-hour race
- `WINDOW_SEC = 1800`, `RACE_SEC = 86400`, `LAP_KM = 0.4`
- `WINDOW_COLS` — list of column names `["window_0", ..., "window_47"]`
- `WINDOW_MIDPOINT_HOURS` — x-axis values for all profile charts
- `DNF_STATUS = "*"` — raceresult.com status for did-not-finish runners

## Prediction Model

The `PaceProfiler` (in `profiles.py`) fits `StandardScaler → PCA(10) → KMeans(3)` on normalised 48-window pace vectors. Profile curves are k-means centroids inverse-transformed to pace space.

Prediction (`predictor.py`) uses two tiers:
- **Tier 1 (<6 observed windows / <3 h)**: constant-pace extrapolation with ±20% CI
- **Tier 2 (≥6 windows)**: OLS scale fitting against each centroid — `s = (obs_abs · curve_obs) / ||curve_obs||²` — picks the best-fitting profile, fills unobserved windows with `s × centroid_curve`

**Do not attempt PCA completion** (fitting PCA coordinates on observed windows then projecting forward). This extrapolation is numerically unstable on a 10-component basis and produces wildly oscillating late-window predictions.

DNF runners are excluded from PCA/k-means training (incomplete vectors distort centroids) but their partial laps can be fed to the predictor.

## Streamlit Notes

- The DB connection must use `check_same_thread=False` — Streamlit runs pages in threads different from the main thread.
- All heavy objects are cached in `app/cache.py` via `@st.cache_resource` (DB connection, profiler) or `@st.cache_data` (DataFrames). Don't load from disk inside page files.
- `st.plotly_chart` doesn't need `use_container_width=True` in Streamlit ≥1.51 — width stretches by default.
