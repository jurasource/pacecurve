import numpy as np

from analysis.features import RACE_SEC, WINDOW_SEC, get_partial_feature_vector
from analysis.profiles import PaceProfiler

# Fewer than this many observed windows → use constant-pace tier 1 fallback
TIER2_MIN_WINDOWS = 6


class Predictor:
    """
    Predicts final 24h race distance from partial lap data.

    Tier 1 (<6 windows observed): constant-pace extrapolation.
    Tier 2 (>=6 windows): PCA completion via PaceProfiler.
    """

    def __init__(self, profiler: PaceProfiler) -> None:
        self.profiler = profiler

    def predict(
        self, laps_observed: list[dict], verbose: bool = False
    ) -> dict:
        """
        laps_observed: list of {'lap_number': int, 'split_time_sec': int/float}

        Returns dict with:
            predicted_km, predicted_laps, confidence_interval (km tuple),
            profile_label, hours_observed, tier.
        """
        if not laps_observed:
            return self._empty_result()

        vector, mean_pace = get_partial_feature_vector(laps_observed)
        obs_windows = int(np.sum(~np.isnan(vector)))
        splits = np.array([float(l["split_time_sec"]) for l in laps_observed])
        elapsed_sec = float(np.sum(splits))
        hours_observed = elapsed_sec / 3600.0

        if obs_windows < TIER2_MIN_WINDOWS:
            # Tier 1: constant pace
            laps_so_far = len(laps_observed)
            pace = elapsed_sec / laps_so_far if laps_so_far > 0 else 300.0
            predicted_laps = RACE_SEC / pace
            predicted_km = predicted_laps * 0.4
            tier = 1
            profile_label = "Unknown"
            # Wide CI for tier 1 (±20% of prediction)
            ci = (predicted_km * 0.80, predicted_km * 1.20)
        else:
            # Tier 2: PCA completion
            norm_vector = vector / mean_pace  # relative (normalised) form
            predicted_km = self.profiler.predict_distance(norm_vector, mean_pace)
            predicted_laps = predicted_km / 0.4
            tier = 2

            # Profile label via cluster assignment on the observed windows
            import pandas as pd
            from analysis.features import N_WINDOWS
            from analysis.profiles import WINDOW_COLS
            # Build a single-row feature df for assignment
            row = {f"window_{i}": vector[i] for i in range(N_WINDOWS)}
            row_norm = {f"window_{i}": norm_vector[i] for i in range(N_WINDOWS)}
            feat_series = pd.DataFrame([row_norm])
            feat_series.index = pd.MultiIndex.from_tuples(
                [(0, "live")], names=["event_id", "pid"]
            )
            profile_label = self.profiler.assign(feat_series).iloc[0]

            # Confidence interval from stored bounds if available, else ±15%
            ci = self._confidence_interval(predicted_km, obs_windows)

        if verbose:
            print(
                f"Tier {tier} | {hours_observed:.1f}h observed | "
                f"profile={profile_label} | predicted={predicted_km:.1f} km"
            )

        return {
            "predicted_km": round(predicted_km, 2),
            "predicted_miles": round(predicted_km * 0.621371, 2),
            "predicted_laps": int(round(predicted_laps)),
            "confidence_interval_km": (round(ci[0], 1), round(ci[1], 1)),
            "profile_label": profile_label,
            "hours_observed": round(hours_observed, 2),
            "tier": tier,
        }

    def predict_trajectory(self, laps_observed: list[dict]) -> dict:
        """
        Returns observed and predicted pace trajectories for charting.

        Returns dict with:
            hours: x-axis values (midpoint of each 30-min window, 0..24h)
            observed_pace: sec/lap, NaN for future windows
            predicted_pace: sec/lap, full 48 windows
        """
        if not laps_observed:
            hours = [(i + 0.5) * WINDOW_SEC / 3600 for i in range(48)]
            return {
                "hours": hours,
                "observed_pace": [np.nan] * 48,
                "predicted_pace": [np.nan] * 48,
            }

        vector, mean_pace = get_partial_feature_vector(laps_observed)
        norm_vector = vector / mean_pace if mean_pace > 0 else vector

        observed_abs, full_abs = self.profiler.predict_trajectory(norm_vector, mean_pace)

        hours = [(i + 0.5) * WINDOW_SEC / 3600 for i in range(48)]
        return {
            "hours": hours,
            "observed_pace": observed_abs.tolist(),
            "predicted_pace": full_abs.tolist(),
        }

    def _confidence_interval(
        self, predicted_km: float, obs_windows: int
    ) -> tuple[float, float]:
        if (
            self.profiler.confidence_bounds_ is not None
            and obs_windows < len(self.profiler.confidence_bounds_)
        ):
            p10, p90 = self.profiler.confidence_bounds_[obs_windows]
            return (predicted_km + p10, predicted_km + p90)
        # Fallback: ±15%
        return (predicted_km * 0.85, predicted_km * 1.15)

    @staticmethod
    def _empty_result() -> dict:
        return {
            "predicted_km": None,
            "predicted_miles": None,
            "predicted_laps": None,
            "confidence_interval_km": None,
            "profile_label": "Unknown",
            "hours_observed": 0.0,
            "tier": 0,
        }
