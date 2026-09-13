from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests

from .shadow_bigdata_features import STARTER_HISTORY, build_live_values

MODEL_PATH = Path("models/shadow_candidate.joblib")


@lru_cache(maxsize=1)
def _artifact():
    if not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


def available() -> bool:
    return _artifact() is not None and STARTER_HISTORY.exists() and STARTER_HISTORY.stat().st_size > 0


def metadata() -> dict:
    a = _artifact() or {}
    keys = (
        "name", "version", "training_rows", "trained_through", "sigma_runs", "sigma_diff",
        "feature_count", "candidate_feature_count", "validation", "feature_policy",
    )
    return {k: a.get(k) for k in keys if a.get(k) is not None}


def _game_daynight(game_pk: Any, timeout: int = 6):
    try:
        r = requests.get(f"https://statsapi.mlb.com/api/v1.1/game/{int(game_pk)}/feed/live", timeout=timeout)
        r.raise_for_status()
        value = str(r.json().get("gameData", {}).get("datetime", {}).get("dayNight") or "").lower()
        return value if value in {"day", "night"} else None
    except Exception:
        return None


def build_live_features(service, game: dict, home_code: str, away_code: str,
                        off_h: float, off_a: float, pit_h: float, pit_a: float, game_date):
    a = _artifact()
    if a is None:
        raise RuntimeError("shadow candidate model artifact unavailable")

    condition = _game_daynight(game.get("game_pk"))
    values = build_live_values(
        service=service,
        game=game,
        home=home_code,
        away=away_code,
        game_date=game_date,
        day_night=condition,
        off_h=off_h,
        off_a=off_a,
        pit_h=pit_h,
        pit_a=pit_a,
    )
    features = list(a.get("features") or [])
    if not features:
        raise RuntimeError("shadow candidate feature contract unavailable")
    X = pd.DataFrame([[values.get(c, np.nan) for c in features]], columns=features)
    return X, condition


def predict(service, game: dict, home_code: str, away_code: str,
            off_h: float, off_a: float, pit_h: float, pit_a: float, game_date):
    a = _artifact()
    if a is None:
        return None
    X, condition = build_live_features(service, game, home_code, away_code, off_h, off_a, pit_h, pit_a, game_date)
    p = float(a["classifier"].predict_proba(X)[0, 1])
    runs = float(a["runs_model"].predict(X)[0])
    diff = float(a["diff_model"].predict(X)[0])
    return {
        "Probabilidad_Local": round(p * 100.0, 2),
        "Probabilidad_Visita": round((1.0 - p) * 100.0, 2),
        "Proyeccion_Carreras": round(runs, 2),
        "Proyeccion_Handicap_Local": round(diff, 2),
        "Sigma_Carreras": float(a.get("sigma_runs", 3.5)),
        "Sigma_Handicap": float(a.get("sigma_diff", 4.2)),
        "Model_Version": str(a.get("version", "shadow-bigdata-v3")),
        "Feature_Count": int(a.get("feature_count") or len(a.get("features") or [])),
        "DayNight": condition,
    }
