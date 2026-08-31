"""Build exact historical MLB starter IDs keyed by GameID.

Uses MLB's official schedule endpoint with probablePitcher hydration. For completed
historical games this gives a stable gamePk -> pitcher-ID mapping without fuzzy name
matching. The backtest still uses only pitcher PERFORMANCE from seasons before the
game; the pitcher ID and throwing hand are identity/biographical fields only.
"""
from __future__ import annotations

import calendar
from pathlib import Path
import time

import pandas as pd
import requests

OUT = Path("data/mlb_game_starters_history.csv")
SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
SEASONS = range(2021, 2027)
TIMEOUT = 30


def _request(params: dict, attempts: int = 3) -> dict:
    headers = {"User-Agent": "MLB-Pred-validation/1.0", "Accept": "application/json"}
    last = None
    for attempt in range(attempts):
        try:
            r = requests.get(SCHEDULE_URL, params=params, headers=headers, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last = exc
            if attempt + 1 < attempts:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"MLB schedule request failed: {last}")


def _months(year: int):
    # MLB games can occur March-November. Monthly chunks keep payloads modest.
    for month in range(3, 12):
        last = calendar.monthrange(year, month)[1]
        yield f"{year}-{month:02d}-01", f"{year}-{month:02d}-{last:02d}"


def fetch_season(year: int) -> pd.DataFrame:
    rows = []
    for start, end in _months(year):
        payload = _request({
            "sportId": 1,
            "startDate": start,
            "endDate": end,
            "gameTypes": "R,P",
            "hydrate": "probablePitcher",
        })
        for date_block in payload.get("dates", []) or []:
            for game in date_block.get("games", []) or []:
                teams = game.get("teams") or {}
                home = teams.get("home") or {}
                away = teams.get("away") or {}
                hp = home.get("probablePitcher") or {}
                ap = away.get("probablePitcher") or {}
                rows.append({
                    "GameID": game.get("gamePk"),
                    "Date": game.get("officialDate") or date_block.get("date"),
                    "Season": int(year),
                    "HomeStarterID": hp.get("id"),
                    "HomeStarterName": hp.get("fullName"),
                    "AwayStarterID": ap.get("id"),
                    "AwayStarterName": ap.get("fullName"),
                })
        time.sleep(0.10)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["GameID"] = pd.to_numeric(frame["GameID"], errors="coerce")
    frame["HomeStarterID"] = pd.to_numeric(frame["HomeStarterID"], errors="coerce")
    frame["AwayStarterID"] = pd.to_numeric(frame["AwayStarterID"], errors="coerce")
    return frame.dropna(subset=["GameID"]).drop_duplicates("GameID", keep="last")


def main():
    parts = []
    for year in SEASONS:
        print(f"Descargando abridores MLB {year}...")
        try:
            part = fetch_season(year)
            if not part.empty:
                home_cov = part["HomeStarterID"].notna().mean()
                away_cov = part["AwayStarterID"].notna().mean()
                print(f"  {year}: games={len(part)} home_id={home_cov:.1%} away_id={away_cov:.1%}")
                parts.append(part)
        except Exception as exc:
            print(f"WARN {year}: {exc}")
    frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if frame.empty:
        raise RuntimeError("No se pudo construir el mapa histórico de abridores")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)
    both = (frame["HomeStarterID"].notna() & frame["AwayStarterID"].notna()).mean()
    print(f"OK: {OUT} rows={len(frame)} both_starters={both:.1%}")


if __name__ == "__main__":
    main()
