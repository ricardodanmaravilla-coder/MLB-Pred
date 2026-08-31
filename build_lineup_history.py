"""Build historical starting lineups from the official MLB live game feed.

Validation-only dataset. Missing lineups are skipped; no player or batting-order
information is fabricated.
"""
from __future__ import annotations

import concurrent.futures as cf
import time
from pathlib import Path

import pandas as pd
import requests

GAMES = Path("data/mlb_games.csv")
OUT = Path("data/mlb_lineup_history.csv")
URL = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"


def _fetch(game_pk: int, date: str):
    for attempt in range(3):
        try:
            r = requests.get(URL.format(game_pk=game_pk), timeout=20)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            j = r.json()
            rows = []
            box = j.get("liveData", {}).get("boxscore", {})
            teams = box.get("teams", {})
            for side in ("away", "home"):
                t = teams.get(side, {}) or {}
                team = (t.get("team") or {}).get("abbreviation") or (t.get("team") or {}).get("name")
                players = t.get("players", {}) or {}
                order = t.get("battingOrder", []) or []
                # battingOrder contains player IDs in official batting order.
                if len(order) < 9:
                    continue
                for slot, pid in enumerate(order[:9], 1):
                    p = players.get(f"ID{pid}", {}) or {}
                    person = p.get("person", {}) or {}
                    pos = p.get("position", {}) or {}
                    rows.append({
                        "GameID": int(game_pk), "Date": date, "Side": side,
                        "Team": team, "BattingOrder": slot, "PlayerID": int(pid),
                        "PlayerName": person.get("fullName"),
                        "Position": pos.get("abbreviation"),
                        "Source": "MLB_OFFICIAL_LIVE_FEED",
                    })
            return rows, None
        except Exception as e:
            if attempt == 2:
                return [], str(e)
            time.sleep(0.5 * (attempt + 1))


def main():
    g = pd.read_csv(GAMES, low_memory=False)
    idcol = next((c for c in ("GameID", "gamePk", "game_id") if c in g.columns), None)
    if not idcol or "Date" not in g.columns:
        raise SystemExit("mlb_games.csv requires GameID/gamePk and Date")
    g["Date"] = pd.to_datetime(g["Date"], errors="coerce")
    g = g[g["Date"].dt.year >= 2025].dropna(subset=["Date", idcol]).copy()
    jobs = [(int(r[idcol]), r["Date"].date().isoformat()) for _, r in g.iterrows()]
    rows, failures = [], []
    print(f"Descargando lineups oficiales para {len(jobs)} juegos MLB desde 2025...")
    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(_fetch, pk, d): pk for pk, d in jobs}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            rr, err = fut.result(); rows.extend(rr)
            if err: failures.append((futs[fut], err))
            if i % 500 == 0 or i == len(jobs):
                print(f"  procesados={i}/{len(jobs)} rows={len(rows)} fallos={len(failures)}")
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    complete = 0 if out.empty else int((out.groupby(["GameID", "Side"]).size() >= 9).groupby(level=0).sum().ge(2).sum())
    coverage = complete / max(len(jobs), 1)
    print(f"OK: {OUT} rows={len(out)} complete_games={complete} coverage={coverage:.1%} failures={len(failures)}")
    if coverage < 0.85:
        raise SystemExit(f"Insufficient official lineup coverage: {coverage:.1%}")


if __name__ == "__main__":
    main()
