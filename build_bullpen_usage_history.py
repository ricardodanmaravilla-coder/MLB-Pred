"""Build leak-safe historical MLB reliever usage from official game feeds.

This dataset is validation-only. It records what each reliever actually did in a
completed game, so downstream backtests can reconstruct bullpen availability using
ONLY games completed before the target game date.

Source: https://statsapi.mlb.com/api/v1.1/game/{gamePk}/feed/live
No missing pitch counts or performance metrics are fabricated.
"""
from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

from modules.historical_mlb import prepare_games
from modules.team_utils import normalize_team

OUT = Path("data/mlb_bullpen_usage_history.csv")
START_SEASON = 2025
MAX_WORKERS = 12
TIMEOUT = 20


def _num(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _innings(v):
    """Convert baseball innings notation (e.g. 1.2 = 1 2/3) to true innings."""
    if v in (None, ""):
        return None
    s = str(v).strip()
    try:
        if "." not in s:
            return float(s)
        whole, frac = s.split(".", 1)
        outs = int(frac[:1] or 0)
        if outs not in (0, 1, 2):
            return None
        return float(int(whole)) + outs / 3.0
    except Exception:
        return None


def _get_json(url: str, attempts: int = 4):
    last = None
    for i in range(attempts):
        try:
            r = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": "MLB-Pred-validation/1.0"})
            if r.status_code == 429:
                time.sleep(1.0 + i * 1.5)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last = exc
            if i + 1 < attempts:
                time.sleep(0.35 * (2 ** i))
    raise RuntimeError(str(last) if last else "request failed")


def _extract_game(game_pk: int):
    data = _get_json(f"https://statsapi.mlb.com/api/v1.1/game/{int(game_pk)}/feed/live")
    gd = data.get("gameData", {}) or {}
    status = (gd.get("status", {}) or {}).get("abstractGameState", "")
    if str(status).lower() != "final":
        return []
    dt = pd.to_datetime((gd.get("datetime", {}) or {}).get("officialDate"), errors="coerce")
    if pd.isna(dt):
        dt = pd.to_datetime((gd.get("datetime", {}) or {}).get("dateTime"), utc=True, errors="coerce")
        if pd.notna(dt):
            dt = dt.tz_convert(None).normalize()
    if pd.isna(dt):
        return []

    live = data.get("liveData", {}) or {}
    box = live.get("boxscore", {}) or {}
    teams_live = box.get("teams", {}) or {}
    teams_gd = gd.get("teams", {}) or {}
    rows = []
    for side in ("home", "away"):
        tbox = teams_live.get(side, {}) or {}
        pitcher_ids = tbox.get("pitchers", []) or []
        if not pitcher_ids:
            continue
        team_abbr = normalize_team(((teams_gd.get(side, {}) or {}).get("abbreviation")))
        players = tbox.get("players", {}) or {}
        starter_id = int(pitcher_ids[0]) if pitcher_ids else None
        for pid_raw in pitcher_ids:
            try:
                pid = int(pid_raw)
            except Exception:
                continue
            if pid == starter_id:
                continue
            p = players.get(f"ID{pid}", {}) or {}
            stats = ((p.get("stats", {}) or {}).get("pitching", {}) or {})
            person = p.get("person", {}) or {}
            ip = _innings(stats.get("inningsPitched"))
            pitches = _num(stats.get("numberOfPitches"))
            if pitches is None:
                pitches = _num(stats.get("pitchesThrown"))
            rows.append({
                "GameID": int(game_pk),
                "Date": pd.Timestamp(dt).date().isoformat(),
                "Team": team_abbr,
                "PitcherID": pid,
                "PitcherName": person.get("fullName"),
                "IP": ip,
                "Pitches": pitches,
                "BF": _num(stats.get("battersFaced")),
                "ER": _num(stats.get("earnedRuns")),
                "H": _num(stats.get("hits")),
                "BB": _num(stats.get("baseOnBalls")),
                "SO": _num(stats.get("strikeOuts")),
                "HR": _num(stats.get("homeRuns")),
                "Source": "MLB_STATSAPI_GAME_FEED",
            })
    return rows


def main():
    games = prepare_games(pd.read_csv("data/mlb_games.csv"))
    if games.empty or "GameID" not in games.columns:
        raise RuntimeError("mlb_games.csv no contiene GameID utilizable")
    games = games.copy()
    games["GameID"] = pd.to_numeric(games["GameID"], errors="coerce")
    games["Season"] = pd.to_numeric(games["Season"], errors="coerce")
    target = games[(games["Season"] >= START_SEASON) & games["GameID"].notna()].copy()
    ids = sorted({int(x) for x in target["GameID"].tolist()})
    if not ids:
        raise RuntimeError("No hay juegos históricos para construir uso de bullpen")

    all_rows = []
    failures = []
    print(f"Descargando uso real de bullpen para {len(ids)} juegos MLB desde {START_SEASON}...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(_extract_game, gid): gid for gid in ids}
        done = 0
        for fut in as_completed(futs):
            gid = futs[fut]
            done += 1
            try:
                all_rows.extend(fut.result())
            except Exception as exc:
                failures.append((gid, str(exc)))
            if done % 500 == 0 or done == len(ids):
                print(f"  procesados={done}/{len(ids)} rows={len(all_rows)} fallos={len(failures)}")

    if not all_rows:
        raise RuntimeError("MLB StatsAPI no produjo uso histórico de relevistas")
    df = pd.DataFrame(all_rows).drop_duplicates(["GameID", "Team", "PitcherID"]).sort_values(["Date", "GameID", "Team", "PitcherID"])
    coverage = 1.0 - len(failures) / max(1, len(ids))
    if coverage < 0.90:
        raise RuntimeError(f"Cobertura insuficiente de game feeds: {coverage:.1%}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    pitch_cov = pd.to_numeric(df["Pitches"], errors="coerce").notna().mean()
    print(f"OK: {OUT} rows={len(df)} games={df['GameID'].nunique()} request_coverage={coverage:.1%} pitch_count_coverage={pitch_cov:.1%}")
    if failures:
        print("Fallos de muestra:", failures[:10])


if __name__ == "__main__":
    main()
