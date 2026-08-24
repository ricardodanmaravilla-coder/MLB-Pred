import os
import pandas as pd
import requests

SEASON = 2026


def _ip_to_outs(value):
    try:
        text = str(value)
        if "." in text:
            inn, outs = text.split(".", 1)
            return int(inn) * 3 + int(outs[:1] or 0)
        return int(float(text)) * 3
    except Exception:
        return 0


def minar_stats_pitchers():
    print("⚾ Extrayendo pitchers reales desde MLB StatsAPI...")
    os.makedirs("data", exist_ok=True)
    url = (
        "https://statsapi.mlb.com/api/v1/stats"
        f"?stats=season&season={SEASON}&group=pitching&playerPool=ALL&limit=2000&sportId=1"
    )
    response = requests.get(url, headers={"User-Agent": "MLB-Pred/2.0"}, timeout=20)
    response.raise_for_status()
    blocks = response.json().get("stats", [])
    if not blocks:
        raise RuntimeError("MLB StatsAPI no devolvió estadísticas de pitchers")

    rows = []
    for split in blocks[0].get("splits", []):
        player = split.get("player", {})
        team = split.get("team", {})
        stat = split.get("stat", {})
        name = player.get("fullName")
        abbr = team.get("abbreviation")
        era = stat.get("era")
        if not name or not abbr or era in (None, "-.--"):
            continue
        try:
            era = float(era)
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

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No se obtuvieron pitchers válidos")

    starters = df[df.GS > 0].copy()
    starters.to_csv("data/mlb_pitching_individual.csv", index=False)

    # Relevista: al menos 5 apariciones y <=25% de sus juegos como abridor.
    rel = df[(df.G >= 5) & (df.GS / df.G.clip(lower=1) <= 0.25) & (df.Outs > 0)].copy()
    bullpen_rows = []
    for team, grp in rel.groupby("Team"):
        weight = grp.Outs.astype(float)
        bullpen_rows.append({
            "Team": team, "Season": SEASON,
            "Bullpen_ERA": float((grp.ERA * weight).sum() / weight.sum()),
            "Bullpen_WHIPP": float((grp.WHIP.fillna(grp.WHIP.mean()) * weight).sum() / weight.sum()) if grp.WHIP.notna().any() else None,
            "Relievers": int(len(grp)), "Bullpen_Outs": int(weight.sum()),
        })
    pd.DataFrame(bullpen_rows).to_csv("data/mlb_bullpen.csv", index=False)
    print(f"✅ {len(starters)} pitchers con aperturas; {len(bullpen_rows)} bullpens agregados")


if __name__ == "__main__":
    minar_stats_pitchers()
