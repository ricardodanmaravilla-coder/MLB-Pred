import datetime
import os
import time

import pandas as pd
import requests

from modules.team_utils import normalize_team

SEASON = datetime.date.today().year
BASE = "https://statsapi.mlb.com/api/v1"
HEADERS = {"User-Agent": "MLB-Pred/2.0"}


def _ip_to_outs(value):
    try:
        text = str(value)
        if "." in text:
            inn, outs = text.split(".", 1)
            return int(inn) * 3 + int(outs[:1] or 0)
        return int(float(text)) * 3
    except Exception:
        return 0


def _active_teams():
    r = requests.get(f"{BASE}/teams", params={"sportIds": 1, "season": SEASON}, headers=HEADERS, timeout=20)
    r.raise_for_status()
    out = []
    for team in r.json().get("teams", []):
        if not team.get("active", True):
            continue
        abbr = normalize_team(team.get("abbreviation") or team.get("name"))
        if team.get("id") and abbr:
            out.append((int(team["id"]), abbr))
    return out


def _team_pitcher_splits(team_id):
    params = {
        "stats": "season", "season": SEASON, "group": "pitching",
        "playerPool": "ALL", "teamId": team_id, "limit": 250,
        "sportIds": 1,
    }
    r = requests.get(f"{BASE}/stats", params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    blocks = r.json().get("stats", [])
    return blocks[0].get("splits", []) if blocks else []


def minar_stats_pitchers():
    print(f"⚾ Extrayendo pitchers MLB {SEASON} equipo por equipo...")
    os.makedirs("data", exist_ok=True)
    teams = _active_teams()
    if len(teams) < 25:
        raise RuntimeError(f"Lista de equipos incompleta: {len(teams)}")

    rows = []
    for team_id, abbr in teams:
        try:
            splits = _team_pitcher_splits(team_id)
            print(f"  {abbr}: {len(splits)} pitchers")
            for split in splits:
                player = split.get("player", {})
                stat = split.get("stat", {})
                name = player.get("fullName")
                era_raw = stat.get("era")
                if not name or era_raw in (None, "-.--"):
                    continue
                try:
                    era = float(era_raw)
                except Exception:
                    continue
                g = int(stat.get("gamesPlayed", stat.get("gamesPitched", 0)) or 0)
                gs = int(stat.get("gamesStarted", 0) or 0)
                ip = stat.get("inningsPitched", "0.0")
                rows.append({
                    "PlayerID": player.get("id"), "Name": name, "Team": abbr,
                    "Season": SEASON, "ERA": era, "GS": gs, "G": g,
                    "IP": ip, "Outs": _ip_to_outs(ip),
                    "WHIP": pd.to_numeric(stat.get("whip"), errors="coerce"),
                    "K": pd.to_numeric(stat.get("strikeOuts"), errors="coerce"),
                    "BB": pd.to_numeric(stat.get("baseOnBalls"), errors="coerce"),
                    "HR": pd.to_numeric(stat.get("homeRuns"), errors="coerce"),
                })
            time.sleep(0.05)
        except Exception as exc:
            print(f"⚠️ {abbr}: {exc}")

    df = pd.DataFrame(rows)
    if df.empty or df.Team.nunique() < 20:
        raise RuntimeError(f"Pitchers insuficientes: {len(df)} filas / {0 if df.empty else df.Team.nunique()} equipos")

    starters = df[(df.GS > 0) & (df.Outs > 0)].copy()
    starters.to_csv("data/mlb_pitching_individual.csv", index=False)

    # Relevista operativo: >=5 apariciones y como máximo 25% de apariciones iniciadas.
    rel = df[(df.G >= 5) & ((df.GS / df.G.clip(lower=1)) <= 0.25) & (df.Outs > 0)].copy()
    bullpen_rows = []
    for team, grp in rel.groupby("Team"):
        weight = grp.Outs.astype(float)
        whip = None
        if grp.WHIP.notna().any():
            valid = grp.WHIP.notna()
            whip = float((grp.loc[valid, "WHIP"] * grp.loc[valid, "Outs"]).sum() / grp.loc[valid, "Outs"].sum())
        bullpen_rows.append({
            "Team": team, "Season": SEASON,
            "Bullpen_ERA": float((grp.ERA * weight).sum() / weight.sum()),
            "Bullpen_WHIP": whip,
            "Relievers": int(len(grp)), "Bullpen_Outs": int(weight.sum()),
        })
    bull = pd.DataFrame(bullpen_rows)
    if bull.Team.nunique() < 20:
        raise RuntimeError(f"Bullpen insuficiente: {bull.Team.nunique()} equipos")
    bull.to_csv("data/mlb_bullpen.csv", index=False)
    print(f"✅ {len(starters)} pitchers con aperturas; {len(bull)} bullpens agregados")


if __name__ == "__main__":
    minar_stats_pitchers()
