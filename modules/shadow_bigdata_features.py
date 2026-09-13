from __future__ import annotations

"""Rich feature contract for the isolated MLB Shadow model.

The goal is to use as much *real, pregame* information as possible without touching
V7. Historical training comes from the leak-safe research Parquet; live inference
reconstructs the same feature names from current repository data plus Shadow-only
rolling history files. Missing inputs remain NaN and are handled by the model's
imputer rather than fabricated.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any
import math

import numpy as np
import pandas as pd

from .bigdata_mlb import LEGACY_ML_COLUMNS
from .team_utils import normalize_team

SHADOW_DIR = Path("data/shadow")
STARTER_HISTORY = SHADOW_DIR / "starter_performance_history.csv"
BULLPEN_HISTORY = SHADOW_DIR / "bullpen_usage_history.csv"
STATCAST_DAILY = SHADOW_DIR / "statcast_team_daily_real.csv"

DN_METRICS = ("era", "whip", "k_pct", "bb_pct", "kbb_pct", "hr9")
DAYNIGHT_FEATURES: list[str] = []
for side in ("home", "away"):
    for metric in DN_METRICS:
        DAYNIGHT_FEATURES += [f"{side}_starter_dn_{metric}", f"{side}_starter_dn_delta_{metric}"]
    DAYNIGHT_FEATURES += [f"{side}_starter_dn_ip", f"{side}_starter_dn_weight"]

TEAM_FEATURES = [
    "home_ops_index", "away_ops_index", "home_wrc_plus", "away_wrc_plus",
    "home_woba", "away_woba", "home_iso", "away_iso",
    "home_bb_pct", "away_bb_pct", "home_k_pct", "away_k_pct",
    "home_hardhit_pct", "away_hardhit_pct", "home_barrel_pct", "away_barrel_pct",
    "home_ev", "away_ev", "home_ops_vs_l", "away_ops_vs_l", "home_ops_vs_r", "away_ops_vs_r",
    "home_team_era", "away_team_era", "home_team_fip", "away_team_fip",
    "home_team_xfip", "away_team_xfip", "home_team_siera", "away_team_siera",
    "home_team_kbb_pct", "away_team_kbb_pct", "home_team_whip", "away_team_whip",
    "home_team_gb_pct", "away_team_gb_pct", "home_team_hr9", "away_team_hr9",
]

STARTER_FEATURES = [
    f"{side}_starter_{metric}"
    for side in ("home", "away")
    for metric in ("era", "fip", "xfip", "xera", "k_pct", "bb_pct", "kbb_pct", "whip", "hr9", "gb_pct")
]

BULLPEN_FEATURES = [
    "home_bullpen_era", "away_bullpen_era", "home_bullpen_fip", "away_bullpen_fip",
    "home_bullpen_whip", "away_bullpen_whip", "home_bullpen_kbb_pct", "away_bullpen_kbb_pct",
    "home_bullpen_pitches_1d", "away_bullpen_pitches_1d",
    "home_bullpen_pitches_3d", "away_bullpen_pitches_3d",
    "home_bullpen_high_leverage_available", "away_bullpen_high_leverage_available",
]

ENV_FEATURES = [
    "park_factor", "park_factor_hr", "altitude_ft", "temperature_f", "wind_mph", "day_game",
]

STATCAST_FEATURES = [
    f"{side}_{metric}"
    for side in ("home", "away")
    for metric in ("xwoba_30d", "xslg_30d", "hardhit_30d", "barrel_30d", "ev_30d")
]

BIGDATA_EXTRA_FEATURES = TEAM_FEATURES + STARTER_FEATURES + BULLPEN_FEATURES + ENV_FEATURES + STATCAST_FEATURES + DAYNIGHT_FEATURES
BIGDATA_FEATURES = list(LEGACY_ML_COLUMNS) + BIGDATA_EXTRA_FEATURES


def _num(value: Any, default=np.nan):
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def _latest_team_row(df: pd.DataFrame, team: str):
    if df is None or df.empty or "Team" not in df.columns:
        return None
    x = df.copy()
    x["_team"] = x["Team"].map(normalize_team)
    x = x[x["_team"] == normalize_team(team)]
    if x.empty:
        return None
    if "Season" in x.columns:
        x["_season"] = pd.to_numeric(x["Season"], errors="coerce")
        x = x.sort_values("_season")
    return x.iloc[-1]


def _metric(row, variants):
    if row is None:
        return np.nan
    for name in variants:
        if name in row.index:
            v = pd.to_numeric(pd.Series([row.get(name)]), errors="coerce").iloc[0]
            if pd.notna(v):
                return float(v)
    return np.nan


def _starter_row(df: pd.DataFrame, pitcher_id: Any, name: str | None, team: str):
    if df is None or df.empty:
        return None
    x = df.copy()
    match = pd.DataFrame()
    if "PlayerID" in x.columns and pitcher_id not in (None, ""):
        ids = pd.to_numeric(x["PlayerID"], errors="coerce")
        try:
            match = x[ids == int(float(pitcher_id))]
        except Exception:
            pass
    if match.empty and name and "Name" in x.columns:
        key = str(name).strip().casefold()
        match = x[x["Name"].astype(str).str.strip().str.casefold() == key]
        if match.empty:
            surname = key.split()[-1] if key else ""
            fb = x[x["Name"].astype(str).str.split().str[-1].str.casefold() == surname]
            if "Team" in fb.columns:
                t = fb[fb["Team"].map(normalize_team) == normalize_team(team)]
                if not t.empty:
                    fb = t
            if not fb.empty and fb["Name"].nunique() == 1:
                match = fb
    if match.empty:
        return None
    if "Team" in match.columns:
        t = match[match["Team"].map(normalize_team) == normalize_team(team)]
        if not t.empty:
            match = t
    if "Season" in match.columns:
        match = match.assign(_season=pd.to_numeric(match["Season"], errors="coerce")).sort_values("_season")
    return match.iloc[-1]


@lru_cache(maxsize=1)
def starter_history() -> pd.DataFrame:
    if not STARTER_HISTORY.exists():
        return pd.DataFrame()
    x = pd.read_csv(STARTER_HISTORY, low_memory=False)
    x["Date"] = pd.to_datetime(x["Date"], errors="coerce").dt.normalize()
    x["PitcherID"] = pd.to_numeric(x["PitcherID"], errors="coerce")
    if "DayNight" in x.columns:
        x["DayNight"] = x["DayNight"].astype(str).str.lower()
    return x.dropna(subset=["Date", "PitcherID"])


def _aggregate_pitching(rows: pd.DataFrame):
    if rows is None or rows.empty:
        return None
    sums = {c: float(pd.to_numeric(rows.get(c), errors="coerce").fillna(0).sum()) for c in ("IP", "ER", "H", "BB", "SO", "HR", "BF")}
    ip, bf = sums["IP"], sums["BF"]
    if ip <= 0 or bf <= 0:
        return None
    kp = 100.0 * sums["SO"] / bf
    bp = 100.0 * sums["BB"] / bf
    return {
        "era": 9.0 * sums["ER"] / ip,
        "whip": (sums["H"] + sums["BB"]) / ip,
        "k_pct": kp,
        "bb_pct": bp,
        "kbb_pct": kp - bp,
        "hr9": 9.0 * sums["HR"] / ip,
        "ip": ip,
    }


def rolling_starter(pitcher_id: Any, target_date, last_n: int = 10):
    h = starter_history()
    if h.empty or pitcher_id in (None, ""):
        return None
    try:
        pid = int(float(pitcher_id))
    except Exception:
        return None
    d = pd.Timestamp(target_date).normalize()
    x = h[(h["PitcherID"] == pid) & (h["Date"] < d)].sort_values("Date").tail(last_n)
    return _aggregate_pitching(x)


def daynight_starter(pitcher_id: Any, target_date, condition: str | None, prior_ip: float = 25.0):
    h = starter_history()
    if h.empty or condition not in {"day", "night"} or pitcher_id in (None, "") or "DayNight" not in h.columns:
        return None
    try:
        pid = int(float(pitcher_id))
    except Exception:
        return None
    d = pd.Timestamp(target_date).normalize()
    x = h[(h["PitcherID"] == pid) & (h["Date"] < d)]
    overall = _aggregate_pitching(x)
    split = _aggregate_pitching(x[x["DayNight"] == condition])
    if not overall or not split or split["ip"] < 5.0:
        return None
    w = split["ip"] / (split["ip"] + prior_ip)
    out = {"ip": split["ip"], "weight": w}
    for m in DN_METRICS:
        out[m] = w * split[m] + (1.0 - w) * overall[m]
        out[f"delta_{m}"] = out[m] - overall[m]
    return out


@lru_cache(maxsize=1)
def bullpen_history() -> pd.DataFrame:
    if not BULLPEN_HISTORY.exists():
        return pd.DataFrame()
    x = pd.read_csv(BULLPEN_HISTORY, low_memory=False)
    x["Date"] = pd.to_datetime(x["Date"], errors="coerce").dt.normalize()
    x["Team"] = x["Team"].map(normalize_team)
    x["Pitches"] = pd.to_numeric(x["Pitches"], errors="coerce")
    if "PitcherID" in x.columns:
        x["PitcherID"] = pd.to_numeric(x["PitcherID"], errors="coerce")
    return x.dropna(subset=["Date", "Team"])


def bullpen_fatigue(team: str, target_date):
    h = bullpen_history()
    if h.empty:
        return {}
    team = normalize_team(team)
    d = pd.Timestamp(target_date).normalize()
    x = h[(h["Team"] == team) & (h["Date"] < d)]
    if x.empty:
        return {}
    p1 = float(x[x["Date"] == d - pd.Timedelta(days=1)]["Pitches"].fillna(0).sum())
    p3 = float(x[(x["Date"] >= d - pd.Timedelta(days=3)) & (x["Date"] < d)]["Pitches"].fillna(0).sum())
    avail = np.nan
    if "PitcherID" in x.columns:
        recent30 = x[(x["Date"] >= d - pd.Timedelta(days=30)) & x["PitcherID"].notna()]
        totals = recent30.groupby("PitcherID")["Pitches"].sum().sort_values(ascending=False)
        if len(totals) >= 2:
            top = list(totals.head(3).index)
            ok = 0
            for pid in top:
                p_1 = float(x[(x["PitcherID"] == pid) & (x["Date"] == d - pd.Timedelta(days=1))]["Pitches"].fillna(0).sum())
                p_2 = p_1 + float(x[(x["PitcherID"] == pid) & (x["Date"] == d - pd.Timedelta(days=2))]["Pitches"].fillna(0).sum())
                if p_1 <= 25.0 and p_2 <= 45.0:
                    ok += 1
            avail = ok / len(top)
    return {"pitches_1d": p1, "pitches_3d": p3, "high_leverage_available": avail}


@lru_cache(maxsize=1)
def statcast_daily() -> pd.DataFrame:
    if not STATCAST_DAILY.exists():
        return pd.DataFrame()
    x = pd.read_csv(STATCAST_DAILY, low_memory=False)
    x["Date"] = pd.to_datetime(x["Date"], errors="coerce").dt.normalize()
    x["Team"] = x["Team"].map(normalize_team)
    return x.dropna(subset=["Date", "Team"])


def statcast_30d(team: str, target_date):
    h = statcast_daily()
    if h.empty:
        return {}
    d = pd.Timestamp(target_date).normalize()
    x = h[(h["Team"] == normalize_team(team)) & (h["Date"] >= d - pd.Timedelta(days=30)) & (h["Date"] < d)]
    if x.empty:
        return {}
    def s(c): return float(pd.to_numeric(x.get(c), errors="coerce").fillna(0).sum()) if c in x.columns else 0.0
    den = s("woba_den"); bbe = s("bbe"); xslg_n = s("xslg_n")
    return {
        "xwoba_30d": s("xwoba_num") / den if den >= 25 else np.nan,
        "xslg_30d": s("xslg_sum") / xslg_n if xslg_n >= 15 else np.nan,
        "hardhit_30d": s("hardhit_n") / bbe if bbe >= 15 else np.nan,
        "barrel_30d": s("barrel_n") / bbe if bbe >= 15 else np.nan,
        "ev_30d": s("ev_sum") / bbe if bbe >= 15 else np.nan,
    }


def build_live_values(service, game: dict, home: str, away: str, game_date, day_night: str | None,
                      off_h: float, off_a: float, pit_h: float, pit_a: float):
    base = service.predictor._feature_row(
        service.predictor.current_history, service.predictor.current_h2h,
        normalize_team(home), normalize_team(away), float(off_h), float(off_a), float(pit_h), float(pit_a),
    )
    values = dict(zip(LEGACY_ML_COLUMNS, base))

    bat_map = {
        "ops_index": ["OPS_Index", "OPS"], "wrc_plus": ["wRC+", "wRC_plus"], "woba": ["wOBA"],
        "iso": ["ISO"], "bb_pct": ["BB%", "BB_pct"], "k_pct": ["K%", "K_pct"],
        "hardhit_pct": ["HardHit%", "HardHit_pct"], "barrel_pct": ["Barrel%", "Barrel_pct"],
        "ev": ["EV"], "ops_vs_l": ["OPS_vs_L"], "ops_vs_r": ["OPS_vs_R"],
    }
    pit_map = {
        "team_era": ["ERA"], "team_fip": ["FIP"], "team_xfip": ["xFIP"], "team_siera": ["SIERA"],
        "team_kbb_pct": ["K-BB%", "KBB%"], "team_whip": ["WHIP"],
        "team_gb_pct": ["GB%", "GB_pct"], "team_hr9": ["HR/9", "HR9"],
    }
    for side, team in (("home", home), ("away", away)):
        br = _latest_team_row(service.batting, team)
        pr = _latest_team_row(service.pitching, team)
        for key, variants in bat_map.items(): values[f"{side}_{key}"] = _metric(br, variants)
        for key, variants in pit_map.items(): values[f"{side}_{key}"] = _metric(pr, variants)

        pid = game.get(f"{side}_pitcher_id")
        name = game.get(f"{side}_pitcher")
        sr = _starter_row(service.pitchers, pid, name, team)
        rolling = rolling_starter(pid, game_date) or {}
        starter_map = {
            "era": ["ERA"], "fip": ["FIP"], "xfip": ["xFIP"], "xera": ["xERA"],
            "k_pct": ["K%", "K_pct"], "bb_pct": ["BB%", "BB_pct"], "kbb_pct": ["K-BB%", "KBB%"],
            "whip": ["WHIP"], "hr9": ["HR/9", "HR9"], "gb_pct": ["GB%", "GB_pct"],
        }
        for key, variants in starter_map.items():
            # Rolling last-10 real starts are preferred for metrics we can calculate.
            values[f"{side}_starter_{key}"] = rolling.get(key, _metric(sr, variants))

        bp = _latest_team_row(service.bullpen, team)
        values[f"{side}_bullpen_era"] = _metric(bp, ["ERA"])
        values[f"{side}_bullpen_fip"] = _metric(bp, ["FIP"])
        values[f"{side}_bullpen_whip"] = _metric(bp, ["WHIP"])
        values[f"{side}_bullpen_kbb_pct"] = _metric(bp, ["K-BB%", "KBB%"])
        fatigue = bullpen_fatigue(team, game_date)
        values[f"{side}_bullpen_pitches_1d"] = fatigue.get("pitches_1d", np.nan)
        values[f"{side}_bullpen_pitches_3d"] = fatigue.get("pitches_3d", np.nan)
        values[f"{side}_bullpen_high_leverage_available"] = fatigue.get("high_leverage_available", np.nan)

        sc = statcast_30d(team, game_date)
        for metric in ("xwoba_30d", "xslg_30d", "hardhit_30d", "barrel_30d", "ev_30d"):
            values[f"{side}_{metric}"] = sc.get(metric, np.nan)

        dn = daynight_starter(pid, game_date, day_night)
        for metric in DN_METRICS:
            values[f"{side}_starter_dn_{metric}"] = np.nan if not dn else dn[metric]
            values[f"{side}_starter_dn_delta_{metric}"] = np.nan if not dn else dn[f"delta_{metric}"]
        values[f"{side}_starter_dn_ip"] = np.nan if not dn else dn["ip"]
        values[f"{side}_starter_dn_weight"] = np.nan if not dn else dn["weight"]

    park = _latest_team_row(service.parks, home)
    values["park_factor"] = _metric(park, ["Park_Factor", "ParkFactor"])
    values["park_factor_hr"] = _metric(park, ["Park_Factor_HR", "HR_Factor"])
    values["altitude_ft"] = _metric(park, ["Altitud", "AltitudeFt", "Altitude_Ft", "Altitude"])
    try:
        temp, wind, _direction, _source = service._weather(game.get("home"), game.get("start_time_utc"))
    except Exception:
        temp = wind = np.nan
    values["temperature_f"] = _num(temp)
    values["wind_mph"] = _num(wind)
    values["day_game"] = 1.0 if day_night == "day" else 0.0 if day_night == "night" else np.nan
    return values
