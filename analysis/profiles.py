from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from analysis.features import N_WINDOWS, RACE_SEC

WINDOW_COLS = [f"window_{i}" for i in range(N_WINDOWS)]

# Labels assigned by inspecting profile shape: flattest curve → "Steady",
# steepest late-race fade → "Aggressive", middle → "Conservative".
# Re-ordered at fit time by slope of back half.
_PROFILE_LABELS = ["Steady", "Conservative", "Aggressive"]


def _window_col(i: int) -> str:
    return f"window_{i}"


class PaceProfiler:
    """
    Extracts pace/fatigue archetypes from historical runner data via PCA + k-means.

    fit() expects the *normalised* feature matrix (from features.normalise_features).
    """

    def __init__(self) -> None:
        self.scaler_: StandardScaler | None = None
        self.pca_: PCA | None = None
        self.kmeans_: KMeans | None = None
        self.profile_curves_: np.ndarray | None = None  # shape (n_clusters, 48)
        self.profile_labels_: list[str] = []
        self.n_clusters_: int = 0
        self.n_components_: int = 0
        # Confidence bounds computed from LOYO backtest residuals
        # shape: (n_obs_windows, 2) storing (p10, p90) error in km
        self.confidence_bounds_: np.ndarray | None = None

    def fit(
        self,
        feature_df: pd.DataFrame,
        n_components: int = 10,
        n_clusters: int = 3,
    ) -> "PaceProfiler":
        X = feature_df[WINDOW_COLS].to_numpy(dtype=float)
        # Drop rows that are all-NaN (shouldn't happen after exclude_dnf but be safe)
        valid = ~np.all(np.isnan(X), axis=1)
        X = X[valid]

        # Impute remaining NaN with column means (late windows of slower runners)
        col_means = np.nanmean(X, axis=0)
        nan_mask = np.isnan(X)
        X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        self.n_components_ = n_components
        self.pca_ = PCA(n_components=n_components, random_state=42)
        X_pca = self.pca_.fit_transform(X_scaled)

        self.n_clusters_ = n_clusters
        self.kmeans_ = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        self.kmeans_.fit(X_pca)

        self._compute_profile_curves()
        self._assign_labels()
        return self

    def _compute_profile_curves(self) -> None:
        """Invert centroids from PCA space back to original normalised feature space."""
        centroids_pca = self.kmeans_.cluster_centers_
        centroids_scaled = self.pca_.inverse_transform(centroids_pca)
        self.profile_curves_ = self.scaler_.inverse_transform(centroids_scaled)

    def _assign_labels(self) -> None:
        """
        Label profiles by the slope of their back half (windows 24–47).
        Steeper positive slope (more fade) → "Aggressive".
        Flattest → "Steady".
        """
        curves = self.profile_curves_
        back_slopes = []
        for i in range(self.n_clusters_):
            back = curves[i, 24:]
            x = np.arange(len(back), dtype=float)
            slope = np.polyfit(x, back, 1)[0]
            back_slopes.append(slope)

        order = np.argsort(back_slopes)  # ascending: flattest first
        labels = ["Steady", "Conservative", "Aggressive"]
        self.profile_labels_ = [""] * self.n_clusters_
        for rank, cluster_idx in enumerate(order):
            self.profile_labels_[cluster_idx] = labels[min(rank, len(labels) - 1)]

    def assign(self, feature_df: pd.DataFrame) -> pd.Series:
        """Return cluster labels for each runner in feature_df."""
        X = self._prepare_X(feature_df)
        X_pca = self.pca_.transform(self.scaler_.transform(X))
        labels_int = self.kmeans_.predict(X_pca)
        return pd.Series(
            [self.profile_labels_[i] for i in labels_int],
            index=feature_df.index,
            name="profile",
        )

    def predict_distance(
        self, partial_vector: np.ndarray, mean_pace: float
    ) -> float:
        """
        Predict final 24h distance (km) from a partial normalised pace vector.

        partial_vector: shape (48,), NaN for unobserved windows.
        mean_pace: runner's mean split_time_sec from observed laps (initial scale estimate).

        Fits each profile centroid's scale to the observed data via OLS, picks the
        best-matching profile, then uses:
        - observed windows: actual observed pace
        - unobserved windows: best_scale × profile_centroid[window]
        """
        _, best_scale, best_curve = self._fit_best_profile(partial_vector, mean_pace)
        full_abs = self._blend_trajectory(partial_vector, mean_pace, best_scale, best_curve)
        return self._count_laps(full_abs) * 0.4

    def _fit_best_profile(
        self, partial_vector: np.ndarray, mean_pace: float
    ) -> tuple[int, float, np.ndarray]:
        """
        For each cluster centroid, fit a scale s so that s × curve[obs] ≈ obs_abs.
        Returns (best_cluster_idx, best_scale, best_curve).

        scale = (obs_abs · curve_obs) / (curve_obs · curve_obs)  — OLS with no intercept.
        """
        obs_idx = np.where(~np.isnan(partial_vector))[0]
        obs_abs = partial_vector[obs_idx] * mean_pace  # absolute sec/lap

        best_idx, best_scale, best_residual = 0, mean_pace, np.inf
        for k in range(self.n_clusters_):
            curve = self.profile_curves_[k]          # normalised, shape (48,)
            curve_obs = curve[obs_idx]
            denom = float(curve_obs @ curve_obs)
            if denom == 0:
                continue
            scale = float(obs_abs @ curve_obs) / denom
            residual = float(np.sum((obs_abs - scale * curve_obs) ** 2))
            if residual < best_residual:
                best_residual = residual
                best_idx = k
                best_scale = scale

        return best_idx, best_scale, self.profile_curves_[best_idx]

    @staticmethod
    def _blend_trajectory(
        partial_vector: np.ndarray,
        mean_pace: float,
        best_scale: float,
        best_curve: np.ndarray,
    ) -> np.ndarray:
        """
        Build full 48-window absolute pace array:
        - Observed windows: actual observed abs pace (partial_vector × mean_pace)
        - Unobserved windows: best_scale × best_curve[window]
        """
        full_abs = best_scale * best_curve
        obs_idx = np.where(~np.isnan(partial_vector))[0]
        full_abs[obs_idx] = partial_vector[obs_idx] * mean_pace
        return full_abs

    def predict_trajectory(
        self, partial_vector: np.ndarray, mean_pace: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (observed_abs, full_abs) — absolute sec/lap for all 48 windows.
        observed_abs has NaN for future windows; full_abs is the complete prediction.
        """
        _, best_scale, best_curve = self._fit_best_profile(partial_vector, mean_pace)
        full_abs = self._blend_trajectory(partial_vector, mean_pace, best_scale, best_curve)
        observed_abs = np.where(
            ~np.isnan(partial_vector), partial_vector * mean_pace, np.nan
        )
        return observed_abs, full_abs

    @staticmethod
    def _count_laps(pace_by_window: np.ndarray) -> float:
        """Simulate laps through 24h given per-window average pace (sec/lap)."""
        remaining = float(RACE_SEC)
        total_laps = 0.0
        for w in range(N_WINDOWS):
            if remaining <= 0:
                break
            pace = pace_by_window[w]
            if np.isnan(pace) or pace <= 0:
                pace = 300.0  # fallback: 5-min lap
            window_budget = min(remaining, 1800.0)
            laps_in_window = window_budget / pace
            total_laps += laps_in_window
            remaining -= window_budget
        return total_laps

    def _prepare_X(self, feature_df: pd.DataFrame) -> np.ndarray:
        X = feature_df[WINDOW_COLS].to_numpy(dtype=float)
        col_means = self.scaler_.mean_
        nan_mask = np.isnan(X)
        X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
        return X

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "PaceProfiler":
        return joblib.load(path)


def select_n_clusters(
    feature_df: pd.DataFrame,
    profiler_template: PaceProfiler | None = None,
    max_k: int = 8,
) -> pd.DataFrame:
    """
    Compute silhouette score and inertia for k=2..max_k on feature_df.
    Returns a DataFrame with columns: k, inertia, silhouette.
    """
    X = feature_df[WINDOW_COLS].to_numpy(dtype=float)
    valid = ~np.all(np.isnan(X), axis=1)
    X = X[valid]
    col_means = np.nanmean(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=10, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    rows = []
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(X_pca)
        sil = silhouette_score(X_pca, labels)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    return pd.DataFrame(rows)
