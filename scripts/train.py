"""
Train the PaceProfiler on all available race data and run the LOYO backtest.

Run from the project root:
    python scripts/train.py

Outputs:
    models/profiler.pkl                — fitted PaceProfiler
    models/backtest_results.parquet    — LOYO backtest long-format results
    models/backtest_metrics.parquet    — aggregated metrics per obs_hours
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib

from analysis.backtest import (
    attach_confidence_bounds,
    compute_backtest_metrics,
    compute_per_profile_metrics,
    run_loyo_backtest,
)
from analysis.data import get_all_laps, get_connection
from analysis.features import build_feature_matrix, normalise_features
from analysis.profiles import PaceProfiler

N_CLUSTERS = 3
N_COMPONENTS = 10
OBS_HOURS = [1, 2, 3, 6, 12, 18]

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def main() -> None:
    print("Connecting to database...")
    conn = get_connection()

    print("Building feature matrix (all years)...")
    all_laps = get_all_laps(conn, exclude_dnf=True)
    feat = build_feature_matrix(all_laps)
    norm = normalise_features(feat)
    print(f"  {len(norm)} runners, {norm.shape[1] - 5} window features")

    print(f"Fitting PaceProfiler (n_clusters={N_CLUSTERS}, n_components={N_COMPONENTS})...")
    profiler = PaceProfiler()
    profiler.fit(norm, n_components=N_COMPONENTS, n_clusters=N_CLUSTERS)
    print(f"  Profile labels: {profiler.profile_labels_}")

    print("Running LOYO backtest...")
    bt = run_loyo_backtest(
        conn,
        n_clusters=N_CLUSTERS,
        n_components=N_COMPONENTS,
        observation_hours=OBS_HOURS,
    )
    print(f"  {len(bt)} prediction records across {bt['year_held_out'].nunique()} years")

    attach_confidence_bounds(profiler, bt)

    metrics = compute_backtest_metrics(bt)
    print("\nBacktest MAE by observation hour:")
    for _, row in metrics.iterrows():
        print(f"  {row['obs_hours']:>4.0f}h: MAE={row['mae_km']:.1f} km  RMSE={row['rmse_km']:.1f} km")

    per_profile = compute_per_profile_metrics(bt)

    profiler_path = MODELS_DIR / "profiler.pkl"
    profiler.save(profiler_path)
    print(f"\nSaved profiler → {profiler_path}")

    bt_path = MODELS_DIR / "backtest_results.parquet"
    bt.to_parquet(bt_path, index=False)
    print(f"Saved backtest results → {bt_path}")

    metrics_path = MODELS_DIR / "backtest_metrics.parquet"
    metrics.to_parquet(metrics_path, index=False)
    print(f"Saved backtest metrics → {metrics_path}")

    per_profile_path = MODELS_DIR / "backtest_per_profile.parquet"
    per_profile.to_parquet(per_profile_path, index=False)
    print(f"Saved per-profile metrics → {per_profile_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
