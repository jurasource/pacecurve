import numpy as np
import pandas as pd

N_WINDOWS = 48
WINDOW_SEC = 1800  # 30 minutes
RACE_SEC = 86400   # 24 hours


def build_feature_matrix(laps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert per-lap data into a (n_runners, 48) feature matrix.

    Each runner becomes one row; the 48 columns are mean split_time_sec
    for each 30-minute window of the 24-hour race. elapsed_sec is
    reconstructed as cumsum(split_time_sec) since the DB column is mostly NULL.

    Returns a DataFrame indexed by (event_id, pid) with columns
    window_0..window_47 plus metadata columns year, name, gender,
    final_distance_km, status.
    """
    df = laps_df.copy()

    # Reconstruct elapsed time
    df = df.sort_values(["event_id", "pid", "lap_number"])
    df["elapsed_sec"] = df.groupby(["event_id", "pid"])["split_time_sec"].cumsum()

    # Assign 30-min window, clip to [0, N_WINDOWS - 1]
    df["window"] = (df["elapsed_sec"] // WINDOW_SEC).clip(0, N_WINDOWS - 1).astype(int)

    # Average pace per window per runner
    window_avg = (
        df.groupby(["event_id", "pid", "window"])["split_time_sec"]
        .mean()
        .reset_index()
    )

    # Pivot to wide format: rows = runners, cols = windows
    pivoted = window_avg.pivot(
        index=["event_id", "pid"], columns="window", values="split_time_sec"
    )
    pivoted.columns = [f"window_{i}" for i in pivoted.columns]
    pivoted = pivoted.reindex(columns=[f"window_{i}" for i in range(N_WINDOWS)])

    # Attach metadata (one row per runner — take first occurrence)
    meta = (
        df.groupby(["event_id", "pid"])[["year", "name", "gender", "final_distance_km", "status"]]
        .first()
    )
    result = pivoted.join(meta)
    return result


def normalise_features(
    feature_df: pd.DataFrame, method: str = "relative"
) -> pd.DataFrame:
    """
    Normalise each runner's 48-window vector so PCA captures fatigue
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
) -> tuple[np.ndarray, float]:
    """
    Build a 48-element array from a partial list of observed laps.

    laps_observed: list of {'lap_number': int, 'split_time_sec': int/float}

    Returns:
        vector: shape (48,), NaN for windows not yet observed
        mean_pace: mean split_time_sec across all observed laps (for de-normalising)
    """
    splits = np.array(
        [float(l["split_time_sec"]) for l in laps_observed], dtype=float
    )
    if len(splits) == 0:
        return np.full(N_WINDOWS, np.nan), np.nan

    elapsed = np.cumsum(splits)
    windows = np.clip(elapsed // WINDOW_SEC, 0, N_WINDOWS - 1).astype(int)

    vector = np.full(N_WINDOWS, np.nan)
    for w in range(N_WINDOWS):
        mask = windows == w
        if mask.any():
            vector[w] = splits[mask].mean()

    mean_pace = splits.mean()
    return vector, mean_pace
