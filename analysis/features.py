import numpy as np
import pandas as pd

N_WINDOWS = 48
WINDOW_SEC = 1800    # 30 minutes per window
RACE_SEC = 86400     # 24-hour race
LAP_KM = 0.4         # standard 400m track lap
KM_TO_MILES = 0.621371

# Midpoint hour of each 30-min window, used for x-axes in charts.
WINDOW_MIDPOINT_HOURS = [(i + 0.5) * WINDOW_SEC / 3600 for i in range(N_WINDOWS)]

DNF_STATUS = "*"     # raceresult.com status value for did-not-finish

WINDOW_COLS = [f"window_{i}" for i in range(N_WINDOWS)]


def _impute_nan_cols(X: np.ndarray, col_means: np.ndarray) -> np.ndarray:
    """Replace NaN entries in X with the corresponding column mean."""
    nan_mask = np.isnan(X)
    if nan_mask.any():
        X = X.copy()
        X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    return X


def build_feature_matrix(laps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert per-lap data into a (n_runners, N_WINDOWS) feature matrix.

    Each runner becomes one row; the N_WINDOWS columns are mean split_time_sec
    for each 30-minute window of the 24-hour race. elapsed_sec is
    reconstructed as cumsum(split_time_sec) since the DB column is mostly NULL.

    Returns a DataFrame indexed by (event_id, pid) with columns
    window_0..window_47 plus metadata columns year, name, gender,
    final_distance_km, status.
    """
    df = laps_df.copy()

    df = df.sort_values(["event_id", "pid", "lap_number"])
    df["elapsed_sec"] = df.groupby(["event_id", "pid"])["split_time_sec"].cumsum()
    df["window"] = (df["elapsed_sec"] // WINDOW_SEC).clip(0, N_WINDOWS - 1).astype(int)

    window_avg = (
        df.groupby(["event_id", "pid", "window"])["split_time_sec"]
        .mean()
        .reset_index()
    )

    pivoted = window_avg.pivot(
        index=["event_id", "pid"], columns="window", values="split_time_sec"
    )
    pivoted.columns = [f"window_{i}" for i in pivoted.columns]
    pivoted = pivoted.reindex(columns=[f"window_{i}" for i in range(N_WINDOWS)])

    meta = (
        df.groupby(["event_id", "pid"])[["year", "name", "gender", "final_distance_km", "status"]]
        .first()
    )
    return pivoted.join(meta)


def normalise_features(
    feature_df: pd.DataFrame, method: str = "relative"
) -> pd.DataFrame:
    """
    Normalise each runner's N_WINDOWS-window vector so PCA captures fatigue
    curve shape rather than absolute speed.

    method='relative': divide each row by its own row-mean (ignoring NaN).
    method='absolute': return unchanged.
    """
    if method == "absolute":
        return feature_df.copy()

    window_cols = [c for c in feature_df.columns if c.startswith("window_")]
    out = feature_df.copy()
    row_means = out[window_cols].mean(axis=1)
    out[window_cols] = out[window_cols].div(row_means, axis=0)
    return out


def get_partial_feature_vector(
    laps_observed: list[dict],
) -> tuple[np.ndarray, float, float]:
    """
    Build a N_WINDOWS-element array from a partial list of observed laps.

    laps_observed: list of {'lap_number': int, 'split_time_sec': int/float}

    Returns:
        vector:      shape (N_WINDOWS,), NaN for windows not yet observed
        mean_pace:   mean split_time_sec across all observed laps
        elapsed_sec: total elapsed seconds across all observed laps
    """
    splits = np.array(
        [float(l["split_time_sec"]) for l in laps_observed], dtype=float
    )
    if len(splits) == 0:
        return np.full(N_WINDOWS, np.nan), np.nan, 0.0

    elapsed = np.cumsum(splits)
    elapsed_sec = float(elapsed[-1])
    windows = np.clip(elapsed // WINDOW_SEC, 0, N_WINDOWS - 1).astype(int)

    # Vectorised window averaging via bincount
    counts = np.bincount(windows, minlength=N_WINDOWS)
    sums = np.bincount(windows, weights=splits, minlength=N_WINDOWS)
    vector = np.full(N_WINDOWS, np.nan)
    observed = counts > 0
    vector[observed] = sums[observed] / counts[observed]

    return vector, float(splits.mean()), elapsed_sec
