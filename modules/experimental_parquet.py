"""Experimental MLB research feature store.

This module is intentionally isolated from production.  Nothing under
``modules/web_service.py`` or ``modules/ml_mlb.py`` imports it.  It creates a
separate Parquet dataset that can carry a much wider set of *pregame* variables
for leak-safe research and walk-forward backtests.

The contract is conservative:
- one row per completed historical game;
- only information available before that game is eligible as a feature;
- current/future season team aggregates are never joined to an older game;
- optional historical sources may be absent; their columns stay NaN and a
  coverage report makes that explicit;
- targets are stored but never used as input features.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import json

import numpy as np
import pandas as pd

from .bigdata_mlb import MLBDataWarehouse
from .historical_mlb import prepare_games
from .team_utils import normalize_team

ROOT = Path("data/experimental")
OUT = ROOT / "mlb_research_features_v1.parquet"
REPORT = ROOT / "mlb_research_features_v1_coverage.json"

# Explicit research contract.  Some columns are populated from the current
# repository today; others are reserved for real historical feeds as those are
# collected.  Keeping them here prevents accidental silent schema drift.
RESEARCH_FEATURES = [
    # chronology / form / matchup
    "home_win5","away_win5","home_win20","away_win20","home_rf5","away_rf5",
    "home_ra5","away_ra5","home_rd5","away_rd5","home_rd20","away_rd20",
    "h2h_home_win","h2h_home_rd","h2h_sample","home_rest_days","away_rest_days",
    # team offense / pitching priors
    "home_ops_index","away_ops_index","home_wrc_plus","away_wrc_plus",
    "home_woba","away_woba","home_iso","away_iso","home_bb_pct","away_bb_pct",
    "home_k_pct","away_k_pct","home_hardhit_pct","away_hardhit_pct",
    "home_barrel_pct","away_barrel_pct","home_ev","away_ev",
    "home_ops_vs_l","away_ops_vs_l","home_ops_vs_r","away_ops_vs_r",
    "home_team_era","away_team_era","home_team_fip","away_team_fip",
    "home_team_xfip","away_team_xfip","home_team_siera","away_team_siera",
    "home_team_kbb_pct","away_team_kbb_pct","home_team_whip","away_team_whip",
    "home_team_gb_pct","away_team_gb_pct","home_team_hr9","away_team_hr9",
    # confirmed starter identity and prior information
    "home_starter_id","away_starter_id","home_starter_hand","away_starter_hand",
    "home_starter_era","away_starter_era","home_starter_fip","away_starter_fip",
    "home_starter_xfip","away_starter_xfip","home_starter_xera","away_starter_xera",
    "home_starter_k_pct","away_starter_k_pct","home_starter_bb_pct","away_starter_bb_pct",
    "home_starter_kbb_pct","away_starter_kbb_pct","home_starter_whip","away_starter_whip",
    "home_starter_hr9","away_starter_hr9","home_starter_gb_pct","away_starter_gb_pct",
    # bullpen pregame state
    "home_bullpen_era","away_bullpen_era","home_bullpen_fip","away_bullpen_fip",
    "home_bullpen_whip","away_bullpen_whip","home_bullpen_kbb_pct","away_bullpen_kbb_pct",
    "home_bullpen_pitches_1d","away_bullpen_pitches_1d","home_bullpen_pitches_3d","away_bullpen_pitches_3d",
    "home_bullpen_high_leverage_available","away_bullpen_high_leverage_available",
    # park / environment
    "park_factor","park_factor_hr","altitude_ft","temperature_f","humidity_pct",
    "wind_mph","wind_out_component","roof_closed","day_game",
    # lineup / availability
    "home_lineup_woba","away_lineup_woba","home_lineup_xwoba","away_lineup_xwoba",
    "home_lineup_iso","away_lineup_iso","home_lineup_hardhit_pct","away_lineup_hardhit_pct",
    "home_lineup_barrel_pct","away_lineup_barrel_pct","home_lineup_missing_war","away_lineup_missing_war",
    # Statcast pregame rolling state
    "home_xwoba_30d","away_xwoba_30d","home_xslg_30d","away_xslg_30d",
    "home_hardhit_30d","away_hardhit_30d","home_barrel_30d","away_barrel_30d",
    "home_ev_30d","away_ev_30d","home_oaa_30d","away_oaa_30d",
    # umpire / travel / injury context
    "umpire_run_factor","umpire_strike_factor","home_travel_miles","away_travel_miles",
    "home_timezone_shift","away_timezone_shift","home_injury_war","away_injury_war",
    # market information known before first pitch
    "close_home_ml","close_away_ml","close_total","close_over_odds","close_under_odds",
    "close_home_runline","close_away_runline","market_home_no_vig","market_total_over_no_vig",
]

META_COLUMNS = ["game_key","Date","Season","Home","Away","gamePk"]
TARGET_COLUMNS = ["target_home_win","target_total_runs","target_run_diff"]


def _read(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p, low_memory=False)
    except Exception:
        return pd.DataFrame()


def _numeric(v):
    return pd.to_numeric(v, errors="coerce")


def _season_team_map(df: pd.DataFrame, column: str):
    if df.empty or column not in df.columns or "Team" not in df.columns or "Season" not in df.columns:
        return {}
    x = df[["Team","Season",column]].copy()
    x["Team"] = x["Team"].map(normalize_team)
    x["Season"] = _numeric(x["Season"])
    x[column] = _numeric(x[column])
    return x.dropna().drop_duplicates(["Team","Season"], keep="last").set_index(["Team","Season"])[column].to_dict()


def _prior_team_value(mapping, team, season, default=np.nan):
    # Prior-season only: protects the historical backtest from same-season
    # aggregate leakage when source CSVs hold full-season statistics.
    return mapping.get((normalize_team(team), int(season)-1), default)


def _pick_col(df: pd.DataFrame, variants: Iterable[str]):
    lower = {str(c).casefold(): c for c in df.columns}
    for v in variants:
        if str(v).casefold() in lower:
            return lower[str(v).casefold()]
    return None


def _starter_lookup(df: pd.DataFrame, player_id, name, team, season):
    if df.empty:
        return None
    x = df
    m = pd.DataFrame()
    pid_col = _pick_col(x, ["PlayerID","player_id","pitcher_id"])
    if pid_col is not None and pd.notna(player_id):
        ids = _numeric(x[pid_col])
        try: m = x[ids == int(float(player_id))]
        except Exception: pass
    if m.empty and name:
        ncol = _pick_col(x, ["Name","Player","player_name"])
        if ncol is not None:
            key = str(name).strip().casefold()
            m = x[x[ncol].astype(str).str.strip().str.casefold() == key]
    if m.empty:
        return None
    if "Team" in m.columns:
        mt = m[m["Team"].map(normalize_team) == normalize_team(team)]
        if not mt.empty: m = mt
    if "Season" in m.columns:
        sy = _numeric(m["Season"])
        m = m[sy <= int(season)-1]
        if m.empty: return None
        m = m.assign(_season=sy.loc[m.index]).sort_values("_season")
    return m.iloc[-1] if not m.empty else None


def _row_metric(row, variants):
    if row is None:
        return np.nan
    for v in variants:
        if v in row.index:
            z = _numeric(pd.Series([row.get(v)])).iloc[0]
            if pd.notna(z): return float(z)
    return np.nan


def _row_text(row, variants):
    if row is None: return np.nan
    for v in variants:
        if v in row.index and pd.notna(row.get(v)):
            return str(row.get(v))
    return np.nan


def build_research_parquet(data_dir: str | Path = "data", out: str | Path = OUT) -> dict:
    data_dir = Path(data_dir); out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    games_raw = _read(data_dir / "mlb_games.csv")
    if games_raw.empty:
        raise RuntimeError("data/mlb_games.csv no disponible")

    # Reuse the already-audited leak-safe chronological core.
    base = MLBDataWarehouse.build_leak_safe_features(games_raw)
    games = prepare_games(games_raw).copy()
    if base.empty or games.empty:
        raise RuntimeError("Histórico MLB insuficiente")

    # Canonical merge key.
    norm = MLBDataWarehouse._normalize_games(games_raw)
    gcols = [c for c in norm.columns if c not in base.columns or c == "game_key"]
    frame = base.merge(norm[gcols], on="game_key", how="left", suffixes=("", "_game"))

    batting = _read(data_dir / "mlb_batting.csv")
    pitching = _read(data_dir / "mlb_pitching.csv")
    starters = _read(data_dir / "mlb_pitching_individual.csv")
    bullpen = _read(data_dir / "mlb_bullpen.csv")
    parks = _read(data_dir / "mlb_park_factors.csv")
    odds = _read(data_dir / "mlb_odds_history.csv")

    # Team prior-season maps.  Variant lists tolerate schema evolution.
    bat_variants = {
        "ops_index":["OPS_Index","OPS"], "wrc_plus":["wRC+","wRC_plus"], "woba":["wOBA"],
        "iso":["ISO"], "bb_pct":["BB%","BB_pct"], "k_pct":["K%","K_pct"],
        "hardhit_pct":["HardHit%","HardHit_pct"], "barrel_pct":["Barrel%","Barrel_pct"], "ev":["EV"],
        "ops_vs_l":["OPS_vs_L"], "ops_vs_r":["OPS_vs_R"],
    }
    pit_variants = {
        "team_era":["ERA"], "team_fip":["FIP"], "team_xfip":["xFIP"], "team_siera":["SIERA"],
        "team_kbb_pct":["K-BB%","KBB%"], "team_whip":["WHIP"], "team_gb_pct":["GB%"], "team_hr9":["HR/9"],
    }
    maps = {}
    for key, variants in bat_variants.items():
        col = _pick_col(batting, variants); maps[key] = _season_team_map(batting, col) if col else {}
    for key, variants in pit_variants.items():
        col = _pick_col(pitching, variants); maps[key] = _season_team_map(pitching, col) if col else {}

    for side, team_col in (("home","Home"),("away","Away")):
        for key in bat_variants:
            frame[f"{side}_{key}"] = [
                _prior_team_value(maps[key], t, s) for t,s in zip(frame[team_col], frame["Season"])
            ]
        for key in pit_variants:
            frame[f"{side}_{key}"] = [
                _prior_team_value(maps[key], t, s) for t,s in zip(frame[team_col], frame["Season"])
            ]

    # Starter prior-season stats if starter identities exist in game history.
    starter_name_cols = {"home": _pick_col(frame,["Home_Starter","HomeStarter","home_starter"]),
                         "away": _pick_col(frame,["Away_Starter","AwayStarter","away_starter"])}
    starter_id_cols = {"home": _pick_col(frame,["HomeStarterID","home_starter_id"]),
                       "away": _pick_col(frame,["AwayStarterID","away_starter_id"])}
    starter_metrics = {
        "era":["ERA"], "fip":["FIP"], "xfip":["xFIP"], "xera":["xERA"],
        "k_pct":["K%","K_pct"], "bb_pct":["BB%","BB_pct"], "kbb_pct":["K-BB%","KBB%"],
        "whip":["WHIP"], "hr9":["HR/9"], "gb_pct":["GB%","GB_pct"],
    }
    for side, team_col in (("home","Home"),("away","Away")):
        rows=[]
        for _, r in frame.iterrows():
            name = r.get(starter_name_cols[side]) if starter_name_cols[side] else None
            pid = r.get(starter_id_cols[side]) if starter_id_cols[side] else None
            rows.append(_starter_lookup(starters,pid,name,r[team_col],r["Season"]))
        frame[f"{side}_starter_id"] = [
            _row_metric(x,["PlayerID","player_id","pitcher_id"]) for x in rows
        ]
        frame[f"{side}_starter_hand"] = [
            _row_text(x,["PitchHand","Throws","Hand"]) for x in rows
        ]
        for key, variants in starter_metrics.items():
            frame[f"{side}_starter_{key}"] = [_row_metric(x,variants) for x in rows]

    # Bullpen team priors.  If only current-season summary exists, use only rows
    # carrying an explicit Season and prior season.  Otherwise leave missing.
    bullpen_metrics = {
        "bullpen_era":["ERA"],"bullpen_fip":["FIP"],"bullpen_whip":["WHIP"],"bullpen_kbb_pct":["K-BB%","KBB%"]
    }
    for key, variants in bullpen_metrics.items():
        col = _pick_col(bullpen,variants)
        mp = _season_team_map(bullpen,col) if col else {}
        for side, team_col in (("home","Home"),("away","Away")):
            frame[f"{side}_{key}"] = [_prior_team_value(mp,t,s) for t,s in zip(frame[team_col],frame["Season"])]

    # Park factors available by home team. Historical park table may not carry
    # season; static factors are context, not outcome-derived labels.
    if not parks.empty and "Team" in parks.columns:
        p = parks.copy(); p["_team"] = p["Team"].map(normalize_team); p = p.drop_duplicates("_team", keep="last").set_index("_team")
        def park_value(team, variants):
            t=normalize_team(team)
            if t not in p.index: return np.nan
            row=p.loc[t]
            return _row_metric(row,variants)
        frame["park_factor"] = [park_value(t,["Park_Factor","ParkFactor"]) for t in frame["Home"]]
        frame["park_factor_hr"] = [park_value(t,["Park_Factor_HR","HR_Factor"]) for t in frame["Home"]]
        frame["altitude_ft"] = [park_value(t,["AltitudeFt","Altitude_Ft","Altitude"]) for t in frame["Home"]]

    # Historical game context when already present in canonical games.
    context_map = {
        "temperature_f":["TempF","TemperatureF"], "humidity_pct":["Humidity","HumidityPct"],
        "wind_mph":["WindMph","WindMPH"], "day_game":["DayNight"], "altitude_ft":["AltitudeFt"],
        "park_factor":["ParkFactor"],
    }
    for dest, variants in context_map.items():
        src=_pick_col(frame,variants)
        if src is None: continue
        if dest == "day_game":
            frame[dest]=frame[src].astype(str).str.casefold().map(lambda x: 1.0 if "day" in x else (0.0 if "night" in x else np.nan))
        else:
            current=_numeric(frame[src])
            if dest in frame.columns: frame[dest]=frame[dest].where(frame[dest].notna(),current)
            else: frame[dest]=current

    # Closing odds: join only by stable identifiers/date+teams when available.
    # We never forward-fill across games.
    if not odds.empty:
        o=odds.copy()
        date_col=_pick_col(o,["game_date","Date","date"]); home_col=_pick_col(o,["home","Home"]); away_col=_pick_col(o,["away","Away"])
        if date_col and home_col and away_col:
            o["_date"]=pd.to_datetime(o[date_col],errors="coerce").dt.normalize(); o["_home"]=o[home_col].map(normalize_team); o["_away"]=o[away_col].map(normalize_team)
            o=o.sort_values(date_col).drop_duplicates(["_date","_home","_away"],keep="last")
            omap={(r._date,r._home,r._away):r for r in o.itertuples(index=False)}
            aliases={
                "close_home_ml":["home_odds","home_ml","cuota_loc"],"close_away_ml":["away_odds","away_ml","cuota_vis"],
                "close_total":["total_line","linea_carreras","total"],"close_over_odds":["over_odds","cuota_over"],
                "close_under_odds":["under_odds","cuota_under"]
            }
            for dest, variants in aliases.items():
                vals=[]
                for _,r in frame.iterrows():
                    key=(pd.Timestamp(r["Date"]).normalize(),normalize_team(r["Home"]),normalize_team(r["Away"]))
                    rr=omap.get(key); vals.append(_row_metric(pd.Series(rr._asdict()) if rr is not None else None,variants))
                frame[dest]=vals

    # Materialize full stable schema, including not-yet-covered real variables.
    for col in RESEARCH_FEATURES:
        if col not in frame.columns: frame[col]=np.nan

    keep=[c for c in META_COLUMNS if c in frame.columns] + RESEARCH_FEATURES + TARGET_COLUMNS
    result=frame[keep].copy().sort_values(["Date","game_key"]).reset_index(drop=True)
    result.to_parquet(out,index=False)

    coverage={c:round(float(result[c].notna().mean()),4) for c in RESEARCH_FEATURES}
    payload={
        "rows":int(len(result)), "columns":int(len(result.columns)), "path":str(out),
        "date_min":None if result.empty else str(pd.to_datetime(result["Date"]).min().date()),
        "date_max":None if result.empty else str(pd.to_datetime(result["Date"]).max().date()),
        "coverage":coverage,
        "eligible_50pct":[c for c,v in coverage.items() if v >= 0.50],
        "eligible_80pct":[c for c,v in coverage.items() if v >= 0.80],
    }
    REPORT.write_text(json.dumps(payload,indent=2,sort_keys=True),encoding="utf-8")
    return payload


if __name__ == "__main__":
    print(json.dumps(build_research_parquet(),indent=2))
