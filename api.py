"""
Lightweight FastAPI wrapper around the PaceCurve prediction model.

Start with:  uvicorn api:app --port 8000
"""
import math
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from analysis.features import (
    LAP_KM,
    RACE_SEC,
    WINDOW_SEC,
    WINDOW_COLS,
    get_partial_feature_vector,
)
from analysis.predictor import Predictor, TIER2_MIN_WINDOWS
from analysis.profiles import PaceProfiler


profiler: PaceProfiler | None = None
predictor: Predictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global profiler, predictor
    profiler = PaceProfiler.load("models/profiler.pkl")
    predictor = Predictor(profiler)
    yield


app = FastAPI(lifespan=lifespan)


class Lap(BaseModel):
    lap_number: int
    split_time_sec: float


class PredictRequest(BaseModel):
    laps: list[Lap]
    race_seconds: float = float(RACE_SEC)


class PredictResponse(BaseModel):
    predicted_km: float | None
    ci_low_km: float | None
    ci_high_km: float | None
    tier: int
    hours_observed: float
    profile_label: str | None


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(body: PredictRequest) -> PredictResponse:
    laps = [{"lap_number": l.lap_number, "split_time_sec": l.split_time_sec} for l in body.laps]

    if not laps:
        return PredictResponse(predicted_km=None, ci_low_km=None, ci_high_km=None, tier=0, hours_observed=0.0, profile_label=None)

    is_standard_24h = abs(body.race_seconds - RACE_SEC) < 1

    if is_standard_24h:
        result = predictor.predict(laps)
        if result["predicted_km"] is None:
            return PredictResponse(predicted_km=None, ci_low_km=None, ci_high_km=None, tier=0, hours_observed=0.0, profile_label=None)
        ci = result["confidence_interval_km"]
        return PredictResponse(
            predicted_km=result["predicted_km"],
            ci_low_km=ci[0],
            ci_high_km=ci[1],
            tier=result["tier"],
            hours_observed=result["hours_observed"],
            profile_label=result["profile_label"] if result["profile_label"] != "Unknown" else None,
        )

    # Non-24h race: sum trajectory windows up to race duration
    traj = predictor.predict_trajectory(laps)
    race_hours = body.race_seconds / 3600
    window_hours = WINDOW_SEC / 3600  # 0.5h per window

    total_km = 0.0
    for h, pace in zip(traj["hours"], traj["predicted_pace"]):
        if pace and not math.isnan(pace) and pace > 0:
            window_start = h - window_hours / 2
            window_end = h + window_hours / 2
            effective_hours = min(window_end, race_hours) - max(window_start, 0.0)
            if effective_hours > 0:
                total_km += (effective_hours * 3600 / pace) * LAP_KM

    vector, mean_pace, elapsed_sec = get_partial_feature_vector(laps)
    obs_windows = int(np.sum(~np.isnan(vector)))
    hours_obs = elapsed_sec / 3600
    tier = 2 if obs_windows >= TIER2_MIN_WINDOWS else 1
    ci_factor_low = 0.85 if tier == 2 else 0.80
    ci_factor_high = 1.15 if tier == 2 else 1.20
    profile_label = None
    if tier == 2 and mean_pace > 0:
        norm_vector = vector / mean_pace
        norm_df = pd.DataFrame([norm_vector], columns=WINDOW_COLS)
        norm_df.index = pd.MultiIndex.from_tuples([(0, "live")], names=["event_id", "pid"])
        label = profiler.assign(norm_df).iloc[0]
        profile_label = label if label != "Unknown" else None

    return PredictResponse(
        predicted_km=round(total_km, 2),
        ci_low_km=round(total_km * ci_factor_low, 1),
        ci_high_km=round(total_km * ci_factor_high, 1),
        tier=tier,
        hours_observed=round(hours_obs, 2),
        profile_label=profile_label,
    )
