import numpy as np
import pandas as pd

from analysis.data import get_all_laps
from analysis.features import build_feature_matrix, normalise_features
from analysis.predictor import Predictor
from analysis.profiles import PaceProfiler

DEFAULT_OBS_HOURS = [1, 2, 3, 6, 12, 18]


def run_loyo_backtest(
    conn,
    n_clusters: int = 3,
    observation_hours: list[float] = DEFAULT_OBS_HOURS,
    n_components: int = 10,
) -> pd.DataFrame:
    """
    Leave-One-Year-Out cross-validation.

    For each year: train PaceProfiler on the other 4 years, then predict
    each non-DNF runner's final distance at multiple observation horizons.

    Note: some runners appear in multiple years, causing mild training/test
    overlap. With only 5 years this is unavoidable; flag in results.

    Returns a long-format DataFrame with columns:
        year_held_out, event_id, pid, name, actual_km,
        obs_hours, predicted_km, error_km, profile_label, tier.
    """
    all_laps = get_all_laps(conn, exclude_dnf=True)
    years = sorted(all_laps["year"].unique())
    records = []

    for held_out_year in years:
        train_laps = all_laps[all_laps["year"] != held_out_year]
        test_laps = all_laps[all_laps["year"] == held_out_year]

        # Fit profiler on training years
        feat_train = build_feature_matrix(train_laps)
        norm_train = normalise_features(feat_train)
        profiler = PaceProfiler()
        profiler.fit(norm_train, n_components=n_components, n_clusters=n_clusters)
        predictor = Predictor(profiler)

        test_runners = (
            test_laps.groupby(["event_id", "pid"])[["name", "final_distance_km"]]
            .first()
            .reset_index()
        )

        for _, runner in test_runners.iterrows():
            eid = runner["event_id"]
            pid = runner["pid"]
            name = runner["name"]
            actual_km = runner["final_distance_km"]
            if pd.isna(actual_km) or actual_km <= 0:
                continue

            runner_laps = test_laps[
                (test_laps["event_id"] == eid) & (test_laps["pid"] == pid)
            ].sort_values("lap_number")

            elapsed_cumsum = runner_laps["split_time_sec"].cumsum()

            for obs_h in observation_hours:
                cutoff_sec = obs_h * 3600
                partial = runner_laps[elapsed_cumsum <= cutoff_sec]
                if partial.empty:
                    continue

                laps_in = partial[["lap_number", "split_time_sec"]].astype(int).to_dict("records")
                result = predictor.predict(laps_in)
                pred_km = result["predicted_km"]
                if pred_km is None:
                    continue

                records.append(
                    {
                        "year_held_out": held_out_year,
                        "event_id": eid,
                        "pid": pid,
                        "name": name,
                        "actual_km": actual_km,
                        "obs_hours": obs_h,
                        "predicted_km": pred_km,
                        "error_km": pred_km - actual_km,
                        "abs_error_km": abs(pred_km - actual_km),
                        "profile_label": result["profile_label"],
                        "tier": result["tier"],
                    }
                )

        print(
            f"Year {held_out_year}: {len(test_runners)} runners tested "
            f"on profiler trained from {sorted(all_laps[all_laps['year'] != held_out_year]['year'].unique())}"
        )

    return pd.DataFrame(records)


def compute_backtest_metrics(backtest_df: pd.DataFrame) -> pd.DataFrame:
    """MAE, RMSE, median error, p10/p90 per observation horizon."""

    def metrics(g: pd.DataFrame) -> pd.Series:
        errors = g["error_km"]
        abs_errors = g["abs_error_km"]
        return pd.Series(
            {
                "n": len(g),
                "mae_km": abs_errors.mean(),
                "rmse_km": float(np.sqrt((errors**2).mean())),
                "median_error_km": errors.median(),
                "p10_error_km": errors.quantile(0.10),
                "p90_error_km": errors.quantile(0.90),
            }
        )

    return backtest_df.groupby("obs_hours").apply(metrics, include_groups=False).reset_index()


def compute_per_profile_metrics(backtest_df: pd.DataFrame) -> pd.DataFrame:
    """Same metrics broken down by (obs_hours, profile_label)."""

    def metrics(g: pd.DataFrame) -> pd.Series:
        errors = g["error_km"]
        abs_errors = g["abs_error_km"]
        return pd.Series(
            {
                "n": len(g),
                "mae_km": abs_errors.mean(),
                "rmse_km": float(np.sqrt((errors**2).mean())),
                "median_error_km": errors.median(),
            }
        )

    return (
        backtest_df.groupby(["obs_hours", "profile_label"])
        .apply(metrics, include_groups=False)
        .reset_index()
    )


def attach_confidence_bounds(
    profiler: PaceProfiler, backtest_df: pd.DataFrame
) -> None:
    """
    Compute per-observation-window percentile error bounds and attach to profiler.

    profiler.confidence_bounds_: np.ndarray of shape (max_obs_windows+1, 2)
    where each row is (p10_error, p90_error) in km for that window count.
    """
    from analysis.features import N_WINDOWS

    bounds = np.zeros((N_WINDOWS, 2))
    for w in range(N_WINDOWS):
        # Use rows where observed_hours corresponds to ~w windows (30-min each)
        min_h = w * 0.5
        max_h = (w + 1) * 0.5
        subset = backtest_df[
            (backtest_df["obs_hours"] >= min_h) & (backtest_df["obs_hours"] < max_h)
        ]
        if len(subset) >= 5:  # need at least 5 samples for meaningful quantiles
            bounds[w] = [subset["error_km"].quantile(0.10), subset["error_km"].quantile(0.90)]
        else:
            bounds[w] = [-20.0, 20.0]  # ±20 km wide fallback when too few samples

    profiler.confidence_bounds_ = bounds
