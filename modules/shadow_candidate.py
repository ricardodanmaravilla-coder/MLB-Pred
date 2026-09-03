from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests

from .bigdata_mlb import LEGACY_ML_COLUMNS
from .team_utils import normalize_team

MODEL_PATH = Path("models/shadow_candidate.joblib")
STARTER_HISTORY_PATH = Path("data/shadow/starter_performance_history.csv")
PRIOR_IP = 25.0
MIN_SPLIT_IP = 5.0
DN_METRICS = ("era", "whip", "k_pct", "bb_pct", "kbb_pct", "hr9")


def _agg(rows: pd.DataFrame):
    if rows is None or rows.empty:
        return None
    sums = {c: float(pd.to_numeric(rows[c], errors="coerce").fillna(0).sum()) for c in ("IP", "ER", "H", "BB", "SO", "HR", "BF")}
    ip, bf = sums["IP"], sums["BF"]
    if ip <= 0 or bf <= 0:
        return None
    k = 100.0 * sums["SO"] / bf
    bb = 100.0 * sums["BB"] / bf
    return {"ip": ip, "era": 9.0*sums["ER"]/ip, "whip": (sums["H"]+sums["BB"])/ip,
            "k_pct": k, "bb_pct": bb, "kbb_pct": k-bb, "hr9": 9.0*sums["HR"]/ip}


def _shrunk_daynight(history: pd.DataFrame, pitcher_id: Any, target_date, condition: str):
    if history.empty or pitcher_id in (None, "") or condition not in {"day", "night"}:
        return None
    try:
        pid = int(float(pitcher_id))
    except Exception:
        return None
    d = pd.Timestamp(target_date).normalize()
    x = history[(history["PitcherID"] == pid) & (history["Date"] < d)]
    if x.empty:
        return None
    overall = _agg(x)
    split = _agg(x[x["DayNight"] == condition])
    if not split or not overall or split["ip"] < MIN_SPLIT_IP:
        return None
    w = split["ip"] / (split["ip"] + PRIOR_IP)
    out = {"ip": split["ip"], "weight": w}
    for m in DN_METRICS:
        out[m] = w*split[m] + (1.0-w)*overall[m]
        out[f"delta_{m}"] = out[m] - overall[m]
    return out


@lru_cache(maxsize=1)
def _artifact():
    if not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


@lru_cache(maxsize=1)
def _starter_history():
    if not STARTER_HISTORY_PATH.exists():
        return pd.DataFrame()
    try:
        x = pd.read_csv(STARTER_HISTORY_PATH, low_memory=False)
        x["Date"] = pd.to_datetime(x["Date"], errors="coerce").dt.normalize()
        x["PitcherID"] = pd.to_numeric(x["PitcherID"], errors="coerce")
        x["DayNight"] = x["DayNight"].astype(str).str.lower()
        return x.dropna(subset=["Date", "PitcherID"])
    except Exception:
        return pd.DataFrame()


def available() -> bool:
    return _artifact() is not None and not _starter_history().empty


def metadata() -> dict:
    a = _artifact() or {}
    return {k: a.get(k) for k in ("name", "version", "training_rows", "trained_through", "sigma_runs", "sigma_diff")}


def _game_daynight(game_pk: Any, timeout: int = 6):
    try:
        r = requests.get(f"https://statsapi.mlb.com/api/v1.1/game/{int(game_pk)}/feed/live", timeout=timeout)
        r.raise_for_status()
        value = str(r.json().get("gameData", {}).get("datetime", {}).get("dayNight") or "").lower()
        return value if value in {"day", "night"} else None
    except Exception:
        return None


def _prior_team_row(df: pd.DataFrame, team: str, season: int):
    if df is None or df.empty or "Team" not in df.columns:
        return None
    x = df.copy()
    x["_team"] = x["Team"].map(normalize_team)
    x = x[x["_team"] == normalize_team(team)]
    if x.empty:
        return None
    if "Season" in x.columns:
        x["_season"] = pd.to_numeric(x["Season"], errors="coerce")
        prior = x[x["_season"] <= int(season)-1].sort_values("_season")
        if not prior.empty:
            return prior.iloc[-1]
    return None


def _metric(row, names):
    if row is None:
        return np.nan
    for c in names:
        if c in row.index:
            v = pd.to_numeric(pd.Series([row.get(c)]), errors="coerce").iloc[0]
            if pd.notna(v):
                return float(v)
    return np.nan


def build_live_features(service, game: dict, home_code: str, away_code: str,
                        off_h: float, off_a: float, pit_h: float, pit_a: float, game_date):
    a = _artifact()
    if a is None:
        raise RuntimeError("shadow candidate model artifact unavailable")
    h, v = normalize_team(home_code), normalize_team(away_code)
    if not h or not v:
        raise RuntimeError("shadow team normalization unavailable")

    base = service.predictor._feature_row(service.predictor.current_history, service.predictor.current_h2h,
                                          h, v, float(off_h), float(off_a), float(pit_h), float(pit_a))
    values = dict(zip(LEGACY_ML_COLUMNS, base))
    season = int(pd.Timestamp(game_date).year)
    bh = _prior_team_row(service.batting, h, season); ba = _prior_team_row(service.batting, v, season)
    ph = _prior_team_row(service.pitching, h, season); pa = _prior_team_row(service.pitching, v, season)
    values.update({
        "home_ops_index": _metric(bh, ["OPS_Index", "OPS"]), "away_ops_index": _metric(ba, ["OPS_Index", "OPS"]),
        "home_wrc_plus": _metric(bh, ["wRC+", "wRC_plus"]), "away_wrc_plus": _metric(ba, ["wRC+", "wRC_plus"]),
        "home_ops_vs_l": _metric(bh, ["OPS_vs_L"]), "away_ops_vs_l": _metric(ba, ["OPS_vs_L"]),
        "home_ops_vs_r": _metric(bh, ["OPS_vs_R"]), "away_ops_vs_r": _metric(ba, ["OPS_vs_R"]),
        "home_team_era": _metric(ph, ["ERA"]), "away_team_era": _metric(pa, ["ERA"]),
        "home_team_xfip": _metric(ph, ["xFIP"]), "away_team_xfip": _metric(pa, ["xFIP"]),
        "home_team_whip": _metric(ph, ["WHIP"]), "away_team_whip": _metric(pa, ["WHIP"]),
    })

    condition = _game_daynight(game.get("game_pk"))
    history = _starter_history()
    for side in ("home", "away"):
        stats = _shrunk_daynight(history, game.get(f"{side}_pitcher_id"), game_date, condition)
        for m in DN_METRICS:
            values[f"{side}_starter_dn_{m}"] = np.nan if not stats else stats[m]
            values[f"{side}_starter_dn_delta_{m}"] = np.nan if not stats else stats[f"delta_{m}"]
        values[f"{side}_starter_dn_ip"] = np.nan if not stats else stats["ip"]
        values[f"{side}_starter_dn_weight"] = np.nan if not stats else stats["weight"]

    features = a["features"]
    return pd.DataFrame([[values.get(c, np.nan) for c in features]], columns=features), condition


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
        "Probabilidad_Local": round(p*100.0, 2), "Probabilidad_Visita": round((1.0-p)*100.0, 2),
        "Proyeccion_Carreras": round(runs, 2), "Proyeccion_Handicap_Local": round(diff, 2),
        "Sigma_Carreras": float(a.get("sigma_runs", 3.5)), "Sigma_Handicap": float(a.get("sigma_diff", 4.2)),
        "Model_Version": str(a.get("version", "shadow-candidate-v1")), "DayNight": condition,
    }
