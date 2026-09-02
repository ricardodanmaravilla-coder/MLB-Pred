"""Build historical MLB starter game logs for the isolated research pipeline.

Uses official MLB completed-game feeds. Each output row contains ONLY the starting
pitcher's performance in that completed game. Downstream code converts these game
logs into expanding/rolling pregame features by shifting history before each target
game, preventing target-game or future leakage.
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

OUT = Path("data/mlb_starter_performance_history.csv")
START_SEASON = 2021
MAX_WORKERS = 16
TIMEOUT = 20


def _num(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _innings(v):
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
            r = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": "MLB-Pred-research/1.0"})
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
    status = str(((gd.get("status") or {}).get("abstractGameState") or "")).lower()
    if status != "final":
        return []
    date = pd.to_datetime(((gd.get("datetime") or {}).get("officialDate")), errors="coerce")
    if pd.isna(date):
        return []
    day_night = str(((gd.get("datetime") or {}).get("dayNight") or "")).strip().lower()
    if day_night not in {"day", "night"}:
        day_night = None

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
        try:
            pid = int(pitcher_ids[0])
        except Exception:
            continue
        player = (tbox.get("players", {}) or {}).get(f"ID{pid}", {}) or {}
        stats = ((player.get("stats", {}) or {}).get("pitching", {}) or {})
        person = player.get("person", {}) or {}
        team = normalize_team(((teams_gd.get(side, {}) or {}).get("abbreviation")))
        rows.append({
            "GameID": int(game_pk),
            "Date": pd.Timestamp(date).date().isoformat(),
            "Season": int(pd.Timestamp(date).year),
            "DayNight": day_night,
            "Team": team,
            "Side": side,
            "PitcherID": pid,
            "PitcherName": person.get("fullName"),
            "IP": _innings(stats.get("inningsPitched")),
            "BF": _num(stats.get("battersFaced")),
            "Pitches": _num(stats.get("numberOfPitches")) or _num(stats.get("pitchesThrown")),
            "ER": _num(stats.get("earnedRuns")),
            "H": _num(stats.get("hits")),
            "BB": _num(stats.get("baseOnBalls")),
            "SO": _num(stats.get("strikeOuts")),
            "HR": _num(stats.get("homeRuns")),
            "HBP": _num(stats.get("hitBatsmen")),
            "Source": "MLB_STATSAPI_GAME_FEED",
        })
    return rows


def main():
    games = prepare_games(pd.read_csv("data/mlb_games.csv", low_memory=False))
    if games.empty or "GameID" not in games.columns:
        raise RuntimeError("mlb_games.csv no contiene GameID utilizable")
    games["GameID"] = pd.to_numeric(games["GameID"], errors="coerce")
    games["Season"] = pd.to_numeric(games["Season"], errors="coerce")
    target = games[(games["Season"] >= START_SEASON) & games["GameID"].notna()].copy()
    ids = sorted({int(x) for x in target["GameID"].tolist()})
    if not ids:
        raise RuntimeError("No hay juegos para construir rendimiento histórico de abridores")

    rows, failures = [], []
    print(f"Descargando rendimiento real de abridores para {len(ids)} juegos MLB desde {START_SEASON}...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(_extract_game, gid): gid for gid in ids}
        for done, fut in enumerate(as_completed(futs), 1):
            gid = futs[fut]
            try:
                rows.extend(fut.result())
            except Exception as exc:
                failures.append((gid, str(exc)))
            if done % 1000 == 0 or done == len(ids):
                print(f"  procesados={done}/{len(ids)} rows={len(rows)} fallos={len(failures)}")

    if not rows:
        raise RuntimeError("MLB StatsAPI no produjo rendimiento histórico de abridores")
    df = pd.DataFrame(rows).drop_duplicates(["GameID", "Side"], keep="last")
    request_cov = 1.0 - len(failures) / max(1, len(ids))
    game_cov = df["GameID"].nunique() / max(1, len(ids))
    dn_cov = df["DayNight"].notna().mean() if "DayNight" in df.columns else 0.0
    if request_cov < 0.95 or game_cov < 0.90:
        raise RuntimeError(f"Cobertura insuficiente: requests={request_cov:.1%}, games={game_cov:.1%}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(["Date", "GameID", "Side"]).to_csv(OUT, index=False)
    print(f"OK: {OUT} rows={len(df)} games={df['GameID'].nunique()} request_coverage={request_cov:.1%} game_coverage={game_cov:.1%} daynight_coverage={dn_cov:.1%}")


if __name__ == "__main__":
    main()
