"""Post-process the isolated research Parquet with real historical context.

Research-only: never imported by production. It fills exact starter identity,
reconstructs bullpen workload from strictly prior games, and derives rolling
starter-quality metrics from each pitcher's completed starts strictly BEFORE the
target game. No target-game/future statistics are used.
"""
from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
import json

import numpy as np
import pandas as pd

from modules.experimental_parquet import OUT, REPORT
from modules.team_utils import normalize_team

STARTERS = Path("data/mlb_game_starters_history.csv")
STARTER_PERF = Path("data/mlb_starter_performance_history.csv")
BULLPEN = Path("data/mlb_bullpen_usage_history.csv")
STARTER_WINDOW = 10


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


def _starter_history() -> pd.DataFrame:
    if not STARTER_PERF.exists():
        return pd.DataFrame()
    x = pd.read_csv(STARTER_PERF, low_memory=False)
    need = {"GameID", "Date", "PitcherID", "IP", "ER", "H", "BB", "SO", "HR", "BF"}
    if not need.issubset(x.columns):
        return pd.DataFrame()
    x["Date"] = pd.to_datetime(x["Date"], errors="coerce").dt.normalize()
    for c in ["GameID", "PitcherID", "IP", "ER", "H", "BB", "SO", "HR", "BF", "HBP"]:
        if c in x.columns:
            x[c] = _num(x[c])
    return x.dropna(subset=["Date", "PitcherID"]).sort_values(["Date", "GameID"])


def _aggregate_starts(rows):
    if not rows:
        return None
    g = pd.DataFrame(list(rows))
    sums = {c: float(pd.to_numeric(g.get(c), errors="coerce").fillna(0).sum()) for c in ["IP", "ER", "H", "BB", "SO", "HR", "BF"]}
    ip, bf = sums["IP"], sums["BF"]
    if ip <= 0 or bf <= 0:
        return None
    era = 9.0 * sums["ER"] / ip
    whip = (sums["H"] + sums["BB"]) / ip
    k_pct = 100.0 * sums["SO"] / bf
    bb_pct = 100.0 * sums["BB"] / bf
    return {
        "era": era,
        "whip": whip,
        "k_pct": k_pct,
        "bb_pct": bb_pct,
        "kbb_pct": k_pct - bb_pct,
        "hr9": 9.0 * sums["HR"] / ip,
    }


def _fill_starter_quality(frame: pd.DataFrame) -> pd.DataFrame:
    hist = _starter_history()
    if hist.empty:
        return frame

    # Process target games chronologically. For each pitcher, maintain only starts
    # whose date is strictly earlier than the target game. Same-day starts are NOT
    # admitted, which also protects doubleheaders from accidental leakage.
    target = frame.copy()
    target["_date"] = pd.to_datetime(target["Date"], errors="coerce").dt.normalize()
    order = target.sort_values(["_date", "gamePk"], kind="mergesort").index.tolist()

    by_pid = defaultdict(list)
    for r in hist.itertuples(index=False):
        by_pid[int(r.PitcherID)].append({
            "Date": pd.Timestamp(r.Date), "GameID": int(r.GameID),
            "IP": r.IP, "ER": r.ER, "H": r.H, "BB": r.BB, "SO": r.SO,
            "HR": r.HR, "BF": r.BF,
        })
    cursors = defaultdict(int)
    windows = defaultdict(lambda: deque(maxlen=STARTER_WINDOW))

    cols = ["era", "whip", "k_pct", "bb_pct", "kbb_pct", "hr9"]
    values = {f"{side}_{metric}": {} for side in ("home_starter", "away_starter") for metric in cols}

    for idx in order:
        d = target.at[idx, "_date"]
        if pd.isna(d):
            continue
        # Update only the pitchers required for this target row, and only with
        # starts from dates strictly before the target date.
        for side, id_col in (("home_starter", "home_starter_id"), ("away_starter", "away_starter_id")):
            pid_raw = target.at[idx, id_col] if id_col in target.columns else np.nan
            if pd.isna(pid_raw):
                continue
            pid = int(float(pid_raw))
            logs = by_pid.get(pid, [])
            cur = cursors[pid]
            while cur < len(logs) and logs[cur]["Date"] < d:
                windows[pid].append(logs[cur])
                cur += 1
            cursors[pid] = cur
            agg = _aggregate_starts(windows[pid])
            if agg:
                for metric in cols:
                    values[f"{side}_{metric}"][idx] = agg[metric]

    for col, mapping in values.items():
        series = pd.Series(mapping, dtype=float)
        # Research rolling metrics supersede the old prior-season lookup where
        # available because they are closer to the actual pregame state.
        if col in target.columns:
            target.loc[series.index, col] = series
        else:
            target[col] = series
    return target.drop(columns=["_date"], errors="ignore")


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
    return u.dropna(subset=["Date", "Team"])


def _team_day_maps(u: pd.DataFrame):
    if u.empty:
        return {}, {}
    daily = u.groupby(["Team", "Date"], as_index=False)["Pitches"].sum(min_count=1)
    pitch_map = {(r.Team, pd.Timestamp(r.Date)): float(r.Pitches) if pd.notna(r.Pitches) else np.nan for r in daily.itertuples(index=False)}
    by_pitcher = {}
    if "PitcherID" in u.columns:
        x = u.copy(); x["PitcherID"] = _num(x["PitcherID"]); x = x.dropna(subset=["PitcherID"])
        for (team, date), g in x.groupby(["Team", "Date"]):
            by_pitcher[(team, pd.Timestamp(date))] = {
                int(pid): float(p) for pid, p in g.groupby("PitcherID")["Pitches"].sum(min_count=1).dropna().items()
            }
    return pitch_map, by_pitcher


def _recent_sum(pitch_map, team, date, days):
    vals = []
    for delta in range(1, int(days) + 1):
        v = pitch_map.get((team, date - pd.Timedelta(days=delta)))
        if v is not None and pd.notna(v):
            vals.append(float(v))
    return float(sum(vals)) if vals else 0.0


def _high_leverage_available(by_pitcher, team, date):
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
        if p1 <= 25.0 and p2 <= 45.0:
            available += 1
    return float(available / len(top))


def _fill_bullpen(frame: pd.DataFrame) -> pd.DataFrame:
    u = _bullpen_daily()
    if u.empty:
        return frame
    pitch_map, by_pitcher = _team_day_maps(u)
    dates = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
    for side, team_col in (("home", "Home"), ("away", "Away")):
        teams = frame[team_col].map(normalize_team)
        p1, p3, avail = [], [], []
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
    frame = _fill_starter_quality(frame)
    frame = _fill_bullpen(frame)
    frame.to_parquet(OUT, index=False)

    coverage = {c: round(float(frame[c].notna().mean()), 4) for c in frame.columns if c not in {"Date", "Home", "Away", "game_key"}}
    payload = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else {}
    payload["coverage_after_context"] = coverage
    payload["context_sources"] = {
        "starter_identity": str(STARTERS),
        "starter_performance": str(STARTER_PERF),
        "starter_rule": f"rolling last {STARTER_WINDOW} completed starts, strictly prior dates",
        "starter_available_metrics": ["ERA", "WHIP", "K%", "BB%", "K-BB%", "HR/9"],
        "starter_unfabricated_missing_metrics": ["FIP", "xFIP", "xERA", "GB%"],
        "bullpen_usage": str(BULLPEN),
        "bullpen_rule": "strictly prior calendar days; no same-day/future usage",
    }
    REPORT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "rows": len(frame),
        "home_starter_id_coverage": coverage.get("home_starter_id", 0),
        "away_starter_id_coverage": coverage.get("away_starter_id", 0),
        "home_starter_era_coverage": coverage.get("home_starter_era", 0),
        "away_starter_era_coverage": coverage.get("away_starter_era", 0),
        "home_starter_kbb_coverage": coverage.get("home_starter_kbb_pct", 0),
        "away_starter_kbb_coverage": coverage.get("away_starter_kbb_pct", 0),
        "home_bullpen_1d_coverage": coverage.get("home_bullpen_pitches_1d", 0),
        "away_bullpen_1d_coverage": coverage.get("away_bullpen_pitches_1d", 0),
        "home_high_leverage_coverage": coverage.get("home_bullpen_high_leverage_available", 0),
        "away_high_leverage_coverage": coverage.get("away_bullpen_high_leverage_available", 0),
    }, indent=2))


if __name__ == "__main__":
    main()
