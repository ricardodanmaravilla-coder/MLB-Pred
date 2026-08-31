"""Build season-by-season individual pitcher metrics for leak-safe validation.

This script intentionally stores one row per pitcher-season.  Backtests only use a
pitcher's metrics from seasons strictly before the game being predicted, preventing
current/future season performance from leaking into historical predictions.
"""
from __future__ import annotations

from pathlib import Path
import time

import pandas as pd
from pybaseball import pitching_stats

from modules.team_utils import normalize_team

OUT = Path("data/mlb_pitching_individual_history.csv")
SEASONS = range(2021, 2027)
KEEP = [
    "Name", "Team", "Season", "ERA", "FIP", "xFIP", "SIERA", "WHIP",
    "K-BB%", "K%", "BB%", "GB%", "HR/9", "WAR", "IP", "GS", "G",
]


def _normalize_percent(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    med = x.dropna().abs().median()
    if pd.notna(med) and med <= 1.0:
        x = x * 100.0
    return x


def fetch_season(season: int) -> pd.DataFrame:
    frame = pitching_stats(int(season), int(season), qual=0, ind=1)
    if frame is None or frame.empty or "Name" not in frame.columns:
        return pd.DataFrame(columns=KEEP)
    out = pd.DataFrame()
    out["Name"] = frame["Name"].astype(str)
    out["Team"] = frame.get("Team", pd.Series(index=frame.index, dtype=object)).map(normalize_team)
    out["Season"] = int(season)
    for col in KEEP:
        if col in {"Name", "Team", "Season"}:
            continue
        out[col] = pd.to_numeric(frame[col], errors="coerce") if col in frame.columns else pd.NA
    for col in ("K-BB%", "K%", "BB%"):
        out[col] = _normalize_percent(out[col])
    out["xFIP_Source"] = "FANGRAPHS_HISTORICAL"
    return out.dropna(subset=["Name"]).drop_duplicates(["Name", "Team", "Season"], keep="last")


def main():
    rows = []
    for season in SEASONS:
        print(f"Descargando pitchers FanGraphs {season}...")
        try:
            rows.append(fetch_season(season))
        except Exception as exc:
            print(f"WARN {season}: {exc}")
        time.sleep(1.0)
    frame = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=KEEP)
    if frame.empty:
        raise RuntimeError("No se pudieron obtener datos históricos de pitchers")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)
    print(f"OK: {OUT} -> {len(frame)} filas, {frame['Season'].nunique()} temporadas")


if __name__ == "__main__":
    main()
