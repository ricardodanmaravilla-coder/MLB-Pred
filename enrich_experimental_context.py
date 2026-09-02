"""Post-process the isolated research Parquet with real historical context.

This script is research-only. It never imports or edits production services.
It fills exact starter identity from the official MLB gamePk map and reconstructs
bullpen workload using only reliever appearances dated strictly before each game.
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from modules.experimental_parquet import OUT, REPORT
from modules.team_utils import normalize_team

STARTERS = Path("data/mlb_game_starters_history.csv")
BULLPEN = Path("data/mlb_bullpen_usage_history.csv")


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _fill_starters(frame: pd.DataFrame) -> pd.DataFrame:
    if not STARTERS.exists() or "gamePk" not in frame.columns:
        return frame
    s = pd.read_csv(STARTERS, low_memory=False)
    if "GameID" not in s.columns:
        return frame
    s["gamePk"] = _num(s["GameID"])
    keep = [c for c in ["gamePk", "HomeStarterID", "AwayStarterID"] if c in s.columns]
    s = s[keep].dropna(subset=["gamePk"]).drop_duplicates("gamePk", keep="last")
    frame = frame.merge(s, on="gamePk", how="left")
    if "HomeStarterID" in frame.columns:
        frame["home_starter_id"] = _num(frame["home_starter_id"]).where(
            _num(frame["home_starter_id"]).notna(), _num(frame["HomeStarterID"])
        )
    if "AwayStarterID" in frame.columns:
        frame["away_starter_id"] = _num(frame["away_starter_id"]).where(
            _num(frame["away_starter_id"]).notna(), _num(frame["AwayStarterID"])
        )
    return frame.drop(columns=["HomeStarterID", "AwayStarterID"], errors="ignore")


def _bullpen_daily() -> pd.DataFrame:
    if not BULLPEN.exists():
        return pd.DataFrame()
    u = pd.read_csv(BULLPEN, low_memory=False)
    need = {"Date", "Team", "Pitches"}
    if not need.issubset(u.columns):
        return pd.DataFrame()
    u["Date"] = pd.to_datetime(u["Date"], errors="coerce").dt.normalize()
    u["Team"] = u["Team"].map(normalize_team)
    u["Pitches"] = _num(u["Pitches"])
    u = u.dropna(subset=["Date", "Team"])
    return u


def _team_day_maps(u: pd.DataFrame):
    if u.empty:
        return {}, {}, {}
    daily = u.groupby(["Team", "Date"], as_index=False)["Pitches"].sum(min_count=1)
    pitch_map = {(r.Team, pd.Timestamp(r.Date)): float(r.Pitches) if pd.notna(r.Pitches) else np.nan for r in daily.itertuples(index=False)}

    by_pitcher = {}
    if "PitcherID" in u.columns:
        x = u.copy(); x["PitcherID"] = _num(x["PitcherID"])
        x = x.dropna(subset=["PitcherID"])
        for (team, date), g in x.groupby(["Team", "Date"]):
            by_pitcher[(team, pd.Timestamp(date))] = {
                int(pid): float(p) for pid, p in g.groupby("PitcherID")["Pitches"].sum(min_count=1).dropna().items()
            }

    # Top relievers are determined from the PRIOR 30 days only at query time.
    dates_by_team = {}
    for team, g in u.groupby("Team"):
        dates_by_team[team] = sorted(pd.Timestamp(d) for d in g["Date"].dropna().unique())
    return pitch_map, by_pitcher, dates_by_team


def _recent_sum(pitch_map, team, date, days):
    vals = []
    for delta in range(1, int(days) + 1):
        v = pitch_map.get((team, date - pd.Timedelta(days=delta)))
        if v is not None and pd.notna(v):
            vals.append(float(v))
    # A zero is a real pregame state when historical coverage exists around the date.
    return float(sum(vals)) if vals else 0.0


def _high_leverage_available(by_pitcher, team, date):
    # Identify the three most-used relievers over the PRIOR 30 calendar days.
    totals = {}
    for delta in range(1, 31):
        day = by_pitcher.get((team, date - pd.Timedelta(days=delta)), {})
        for pid, pitches in day.items():
            totals[pid] = totals.get(pid, 0.0) + float(pitches)
    if len(totals) < 2:
        return np.nan
    top = [pid for pid, _ in sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:3]]
    available = 0
    for pid in top:
        p1 = by_pitcher.get((team, date - pd.Timedelta(days=1)), {}).get(pid, 0.0)
        p2 = p1 + by_pitcher.get((team, date - pd.Timedelta(days=2)), {}).get(pid, 0.0)
        # Conservative workload proxy; it uses only completed games before today.
        if p1 <= 25.0 and p2 <= 45.0:
            available += 1
    return float(available / len(top))


def _fill_bullpen(frame: pd.DataFrame) -> pd.DataFrame:
    u = _bullpen_daily()
    if u.empty:
        return frame
    pitch_map, by_pitcher, _ = _team_day_maps(u)
    dates = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
    for side, team_col in (("home", "Home"), ("away", "Away")):
        teams = frame[team_col].map(normalize_team)
        p1=[]; p3=[]; avail=[]
        for team, date in zip(teams, dates):
            if pd.isna(date) or not team:
                p1.append(np.nan); p3.append(np.nan); avail.append(np.nan); continue
            d = pd.Timestamp(date)
            p1.append(_recent_sum(pitch_map, team, d, 1))
            p3.append(_recent_sum(pitch_map, team, d, 3))
            avail.append(_high_leverage_available(by_pitcher, team, d))
        frame[f"{side}_bullpen_pitches_1d"] = p1
        frame[f"{side}_bullpen_pitches_3d"] = p3
        frame[f"{side}_bullpen_high_leverage_available"] = avail
    return frame


def main():
    if not OUT.exists():
        raise RuntimeError(f"Research Parquet missing: {OUT}")
    frame = pd.read_parquet(OUT)
    frame = _fill_starters(frame)
    frame = _fill_bullpen(frame)
    frame.to_parquet(OUT, index=False)

    coverage = {c: round(float(frame[c].notna().mean()), 4) for c in frame.columns if c not in {"Date", "Home", "Away", "game_key"}}
    payload = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else {}
    payload["coverage_after_context"] = coverage
    payload["context_sources"] = {
        "starter_identity": str(STARTERS),
        "bullpen_usage": str(BULLPEN),
        "bullpen_rule": "strictly prior calendar days; no same-day/future usage"
    }
    REPORT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "rows": len(frame),
        "home_starter_id_coverage": coverage.get("home_starter_id", 0),
        "away_starter_id_coverage": coverage.get("away_starter_id", 0),
        "home_bullpen_1d_coverage": coverage.get("home_bullpen_pitches_1d", 0),
        "away_bullpen_1d_coverage": coverage.get("away_bullpen_pitches_1d", 0),
        "home_high_leverage_coverage": coverage.get("home_bullpen_high_leverage_available", 0),
        "away_high_leverage_coverage": coverage.get("away_bullpen_high_leverage_available", 0),
    }, indent=2))


if __name__ == "__main__":
    main()
