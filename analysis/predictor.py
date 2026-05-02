import numpy as np
import pandas as pd

from analysis.features import (
    KM_TO_MILES,
    LAP_KM,
    N_WINDOWS,
    RACE_SEC,
    WINDOW_COLS,
    WINDOW_MIDPOINT_HOURS,
    get_partial_feature_vector,
)
from analysis.profiles import PaceProfiler

# Fewer than this many observed windows → constant-pace tier-1 fallback.
# 6 windows = 3 hours of data; below this, profile fitting is unreliable.
TIER2_MIN_WINDOWS = 6

# Tier-1 CI: ±20% — wide because constant-pace assumption degrades quickly.
_TIER1_CI_LOW = 0.80
_TIER1_CI_HIGH = 1.20

# Tier-2 fallback CI when no backtest bounds are stored: ±15%.
_TIER2_CI_LOW = 0.85
_TIER2_CI_HIGH = 1.15


class Predictor:
    """
    Predicts final 24h race distance from partial lap data.

    Tier 1 (<TIER2_MIN_WINDOWS windows observed): constant-pace extrapolation.
    Tier 2 (>=TIER2_MIN_WINDOWS windows): profile-matching via PaceProfiler.
    """

    def __init__(self, profiler: PaceProfiler) -> None:
        self.profiler = profiler

    def predict(
        self, laps_observed: list[dict], verbose: bool = False
    ) -> dict:
        """
        laps_observed: list of {'lap_number': int, 'split_time_sec': int/float}

        Returns dict with:
            predicted_km, predicted_miles, predicted_laps,
            confidence_interval_km, profile_label, hours_observed, tier.
        """
        if not laps_observed:
            return self._empty_result()

        vector, mean_pace, elapsed_sec = get_partial_feature_vector(laps_observed)
        obs_windows = int(np.sum(~np.isnan(vector)))
        hours_observed = elapsed_sec / 3600.0

        if obs_windows < TIER2_MIN_WINDOWS:
            laps_so_far = len(laps_observed)
            pace = elapsed_sec / laps_so_far if laps_so_far > 0 else 300.0
            predicted_km = (RACE_SEC / pace) * LAP_KM
            tier = 1
            profile_label = "Unknown"
            ci = (predicted_km * _TIER1_CI_LOW, predicted_km * _TIER1_CI_HIGH)
        else:
            norm_vector = vector / mean_pace
            predicted_km = self.profiler.predict_distance(norm_vector, mean_pace)
            tier = 2

            norm_df = pd.DataFrame([norm_vector], columns=WINDOW_COLS)
            norm_df.index = pd.MultiIndex.from_tuples(
                [(0, "live")], names=["event_id", "pid"]
            )
            profile_label = self.profiler.assign(norm_df).iloc[0]
            ci = self._confidence_interval(predicted_km, obs_windows)

        predicted_laps = predicted_km / LAP_KM

        if verbose:
            print(
                f"Tier {tier} | {hours_observed:.1f}h observed | "
                f"profile={profile_label} | predicted={predicted_km:.1f} km"
            )

        return {
            "predicted_km": round(predicted_km, 2),
            "predicted_miles": round(predicted_km * KM_TO_MILES, 2),
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
            hours: midpoint hour of each 30-min window (0..24h)
            observed_pace: sec/lap, NaN for future windows
            predicted_pace: sec/lap, full N_WINDOWS windows
        """
        if not laps_observed:
            return {
                "hours": WINDOW_MIDPOINT_HOURS,
                "observed_pace": [np.nan] * N_WINDOWS,
                "predicted_pace": [np.nan] * N_WINDOWS,
            }

        vector, mean_pace, _ = get_partial_feature_vector(laps_observed)
        norm_vector = vector / mean_pace if mean_pace > 0 else vector

        observed_abs, full_abs = self.profiler.predict_trajectory(norm_vector, mean_pace)

        return {
            "hours": WINDOW_MIDPOINT_HOURS,
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
        return (predicted_km * _TIER2_CI_LOW, predicted_km * _TIER2_CI_HIGH)

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
