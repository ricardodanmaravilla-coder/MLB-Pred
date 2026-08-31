"""Build historical starting lineups and hitter game logs from official MLB feeds.

Validation-only datasets. Missing lineups are skipped and batting statistics are taken
verbatim from completed-game boxscores. No player metric is fabricated. The hitter
log is later aggregated using only games strictly before each prediction date.
"""
from __future__ import annotations

import concurrent.futures as cf
import time
from pathlib import Path

import pandas as pd
import requests

GAMES = Path("data/mlb_games.csv")
LINEUPS_OUT = Path("data/mlb_lineup_history.csv")
HITTERS_OUT = Path("data/mlb_hitter_game_history.csv")
URL = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"

BAT_KEYS = {
    "atBats": "AB", "hits": "H", "doubles": "2B", "triples": "3B",
    "homeRuns": "HR", "baseOnBalls": "BB", "strikeOuts": "SO",
    "hitByPitch": "HBP", "sacFlies": "SF",
}


def _fetch(game_pk: int, date: str):
    for attempt in range(3):
        try:
            r = requests.get(URL.format(game_pk=game_pk), timeout=20)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            j = r.json()
            status = ((j.get("gameData") or {}).get("status") or {}).get("abstractGameState")
            if status not in ("Final", None):
                return [], [], None
            lineup_rows, hitter_rows = [], []
            box = j.get("liveData", {}).get("boxscore", {})
            teams = box.get("teams", {})
            for side in ("away", "home"):
                t = teams.get(side, {}) or {}
                team = (t.get("team") or {}).get("abbreviation") or (t.get("team") or {}).get("name")
                players = t.get("players", {}) or {}
                order = t.get("battingOrder", []) or []
                if len(order) >= 9:
                    for slot, pid in enumerate(order[:9], 1):
                        p = players.get(f"ID{pid}", {}) or {}
                        person = p.get("person", {}) or {}
                        pos = p.get("position", {}) or {}
                        lineup_rows.append({
                            "GameID": int(game_pk), "Date": date, "Side": side,
                            "Team": team, "BattingOrder": slot, "PlayerID": int(pid),
                            "PlayerName": person.get("fullName"),
                            "Position": pos.get("abbreviation"),
                            "Source": "MLB_OFFICIAL_LIVE_FEED",
                        })
                for key, p in players.items():
                    person = p.get("person", {}) or {}
                    pid = person.get("id")
                    batting = ((p.get("stats") or {}).get("batting") or {})
                    if pid is None or not batting:
                        continue
                    vals = {out: batting.get(src) for src, out in BAT_KEYS.items()}
                    ab = pd.to_numeric(vals.get("AB"), errors="coerce")
                    bb = pd.to_numeric(vals.get("BB"), errors="coerce")
                    hbp = pd.to_numeric(vals.get("HBP"), errors="coerce")
                    sf = pd.to_numeric(vals.get("SF"), errors="coerce")
                    pa = sum(float(x) if pd.notna(x) else 0.0 for x in (ab, bb, hbp, sf))
                    if pa <= 0:
                        continue
                    hitter_rows.append({
                        "GameID": int(game_pk), "Date": date, "Side": side, "Team": team,
                        "PlayerID": int(pid), "PlayerName": person.get("fullName"),
                        **vals, "Source": "MLB_OFFICIAL_LIVE_FEED_BOXSCORE",
                    })
            return lineup_rows, hitter_rows, None
        except Exception as e:
            if attempt == 2:
                return [], [], str(e)
            time.sleep(0.5 * (attempt + 1))


def main():
    g = pd.read_csv(GAMES, low_memory=False)
    idcol = next((c for c in ("GameID", "gamePk", "game_id") if c in g.columns), None)
    if not idcol or "Date" not in g.columns:
        raise SystemExit("mlb_games.csv requires GameID/gamePk and Date")
    g["Date"] = pd.to_datetime(g["Date"], errors="coerce")
    g = g[g["Date"].dt.year >= 2025].dropna(subset=["Date", idcol]).copy()
    jobs = [(int(r[idcol]), r["Date"].date().isoformat()) for _, r in g.iterrows()]
    lineup_rows, hitter_rows, failures = [], [], []
    print(f"Descargando lineups y bateo oficial para {len(jobs)} juegos MLB desde 2025...")
    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(_fetch, pk, d): pk for pk, d in jobs}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            lr, hr, err = fut.result(); lineup_rows.extend(lr); hitter_rows.extend(hr)
            if err: failures.append((futs[fut], err))
            if i % 500 == 0 or i == len(jobs):
                print(f"  procesados={i}/{len(jobs)} lineups={len(lineup_rows)} hitter_rows={len(hitter_rows)} fallos={len(failures)}")
    lineups = pd.DataFrame(lineup_rows)
    hitters = pd.DataFrame(hitter_rows)
    LINEUPS_OUT.parent.mkdir(parents=True, exist_ok=True)
    lineups.to_csv(LINEUPS_OUT, index=False)
    hitters.to_csv(HITTERS_OUT, index=False)
    complete = 0 if lineups.empty else int((lineups.groupby(["GameID", "Side"]).size() >= 9).groupby(level=0).sum().ge(2).sum())
    coverage = complete / max(len(jobs), 1)
    hitter_games = 0 if hitters.empty else int(hitters.GameID.nunique())
    print(f"OK: {LINEUPS_OUT} rows={len(lineups)} complete_games={complete} coverage={coverage:.1%}")
    print(f"OK: {HITTERS_OUT} rows={len(hitters)} games={hitter_games} failures={len(failures)}")
    if coverage < 0.85:
        raise SystemExit(f"Insufficient official lineup coverage: {coverage:.1%}")


if __name__ == "__main__":
    main()
