"""Build leak-safe season-by-season pitcher metrics from official MLB StatsAPI.

FanGraphs/pybaseball can reject GitHub-hosted runners with HTTP 403.  This builder
therefore uses MLB's public StatsAPI as the primary historical source.  It stores
one row per pitcher-season and never needs current/future season performance when
backtesting a past game.

Advanced fields that MLB StatsAPI does not publish directly (xFIP/SIERA) are left
missing rather than fabricated.  FIP is a transparent component approximation used
only as a relative historical signal and is explicitly tagged as such.
"""
from __future__ import annotations

from pathlib import Path
import math
import time
from typing import Any

import pandas as pd
import requests

from modules.team_utils import normalize_team

OUT = Path("data/mlb_pitching_individual_history.csv")
SEASONS = range(2021, 2027)
STATS_URL = "https://statsapi.mlb.com/api/v1/stats"
PEOPLE_URL = "https://statsapi.mlb.com/api/v1/people"
TIMEOUT = 30

COLUMNS = [
    "PlayerID", "Name", "Team", "Season", "PitchHand", "ERA", "FIP", "xFIP",
    "SIERA", "WHIP", "K-BB%", "K%", "BB%", "GB%", "HR/9", "WAR", "IP",
    "GS", "G", "BF", "FIP_Source", "Data_Source",
]


def _num(v: Any):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _ip(v: Any):
    """Convert baseball innings notation (e.g. 123.2) into true innings."""
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        if "." not in s:
            return float(int(s))
        whole, frac = s.split(".", 1)
        outs = int(frac[:1] or 0)
        if outs not in (0, 1, 2):
            return float(s)
        return float(int(whole)) + outs / 3.0
    except Exception:
        return _num(v)


def _request_json(url: str, params: dict, attempts: int = 3) -> dict:
    headers = {"User-Agent": "MLB-Pred-validation/1.0", "Accept": "application/json"}
    last = None
    for attempt in range(attempts):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last = exc
            if attempt + 1 < attempts:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"MLB StatsAPI request failed: {last}")


def _pitch_hands(ids: list[int]) -> dict[int, str]:
    """Resolve throwing hand in batches. Missing biography data remains neutral."""
    out: dict[int, str] = {}
    unique = sorted({int(x) for x in ids if x})
    for start in range(0, len(unique), 100):
        batch = unique[start:start + 100]
        try:
            payload = _request_json(PEOPLE_URL, {"personIds": ",".join(map(str, batch))}, attempts=2)
            for p in payload.get("people", []) or []:
                pid = p.get("id")
                hand = str((p.get("pitchHand") or {}).get("code") or "").upper()[:1]
                if pid and hand in {"L", "R"}:
                    out[int(pid)] = hand
        except Exception as exc:
            print(f"WARN pitch-hand batch {start}: {exc}")
        time.sleep(0.10)
    return out


def fetch_season(season: int) -> pd.DataFrame:
    payload = _request_json(
        STATS_URL,
        {
            "stats": "season",
            "group": "pitching",
            "season": int(season),
            "sportIds": 1,
            "playerPool": "ALL",
            "limit": 5000,
            "offset": 0,
        },
    )
    splits = []
    for block in payload.get("stats", []) or []:
        splits.extend(block.get("splits", []) or [])
    rows = []
    for split in splits:
        player = split.get("player") or {}
        team = split.get("team") or {}
        stat = split.get("stat") or {}
        pid = player.get("id")
        name = player.get("fullName")
        if not pid or not name:
            continue
        innings = _ip(stat.get("inningsPitched"))
        k = _num(stat.get("strikeOuts")) or 0.0
        bb = _num(stat.get("baseOnBalls")) or 0.0
        hbp = _num(stat.get("hitBatsmen")) or 0.0
        hr = _num(stat.get("homeRuns")) or 0.0
        bf = _num(stat.get("battersFaced"))
        go = _num(stat.get("groundOuts"))
        ao = _num(stat.get("airOuts"))
        k_pct = (100.0 * k / bf) if bf and bf > 0 else None
        bb_pct = (100.0 * bb / bf) if bf and bf > 0 else None
        kbb_pct = (k_pct - bb_pct) if k_pct is not None and bb_pct is not None else None
        hr9 = (9.0 * hr / innings) if innings and innings > 0 else None
        gb = (100.0 * go / (go + ao)) if go is not None and ao is not None and (go + ao) > 0 else None
        # Relative FIP proxy. A fixed 3.10 constant does not claim exact seasonal
        # FanGraphs FIP; it preserves the component relationship without leakage.
        fip = ((13.0 * hr + 3.0 * (bb + hbp) - 2.0 * k) / innings + 3.10) if innings and innings > 0 else None
        rows.append(
            {
                "PlayerID": int(pid),
                "Name": str(name),
                "Team": normalize_team(team.get("name") or team.get("abbreviation") or ""),
                "Season": int(season),
                "PitchHand": None,
                "ERA": _num(stat.get("era")),
                "FIP": fip,
                "xFIP": None,
                "SIERA": None,
                "WHIP": _num(stat.get("whip")),
                "K-BB%": kbb_pct,
                "K%": k_pct,
                "BB%": bb_pct,
                "GB%": gb,
                "HR/9": hr9,
                "WAR": None,
                "IP": innings,
                "GS": _num(stat.get("gamesStarted")),
                "G": _num(stat.get("gamesPlayed") or stat.get("gamesPitched")),
                "BF": bf,
                "FIP_Source": "MLB_STATSAPI_COMPONENT_APPROX",
                "Data_Source": "MLB_STATSAPI",
            }
        )
    frame = pd.DataFrame(rows, columns=COLUMNS)
    if frame.empty:
        return frame
    hands = _pitch_hands(frame["PlayerID"].dropna().astype(int).tolist())
    frame["PitchHand"] = frame["PlayerID"].map(hands)
    # Require meaningful season workload for profile population but retain all rows;
    # the backtest decides coverage game by game.
    return frame.drop_duplicates(["PlayerID", "Season"], keep="last")


def main():
    rows = []
    for season in SEASONS:
        print(f"Descargando pitchers MLB StatsAPI {season}...")
        try:
            part = fetch_season(season)
            print(f"  {season}: {len(part)} pitchers")
            if not part.empty:
                rows.append(part)
        except Exception as exc:
            print(f"WARN {season}: {exc}")
        time.sleep(0.25)
    frame = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=COLUMNS)
    if frame.empty:
        raise RuntimeError("No se pudieron obtener datos históricos de pitchers desde MLB StatsAPI")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)
    seasons = int(frame["Season"].nunique()) if "Season" in frame else 0
    hand_cov = float(frame["PitchHand"].isin(["L", "R"]).mean()) if len(frame) else 0.0
    print(f"OK: {OUT} -> {len(frame)} filas, {seasons} temporadas, hand_coverage={hand_cov:.1%}")


if __name__ == "__main__":
    main()
