import calendar
import os
import time
from datetime import date, timedelta

import pandas as pd
import statsapi

from modules.team_utils import normalize_team

TEMPORADAS = [2021, 2022, 2023, 2024, 2025, 2026]


def extraer_estadisticas_oficiales_mlb():
    """Descarga métricas que MLB StatsAPI realmente entrega.

    V2 elimina las columnas falsas wRC+ y xFIP. OPS se conserva como OPS y ERA
    como ERA. Métricas Statcast/FIP/xFIP pueden añadirse después desde una fuente
    que realmente las proporcione.
    """
    print("⚾ Extrayendo estadísticas oficiales MLB...")
    os.makedirs("data", exist_ok=True)
    batting, pitching = [], []

    for year in TEMPORADAS:
        try:
            teams = statsapi.get("teams", {"season": year, "sportId": 1}).get("teams", [])
            for team in teams:
                if not team.get("active", True):
                    continue
                team_id = team.get("id")
                abbr = normalize_team(team.get("abbreviation"))
                if not team_id or not abbr:
                    continue

                b = statsapi.get("team_stats", {
                    "teamId": team_id, "season": year, "group": "hitting", "stats": "season"
                })
                splits = b.get("stats", [{}])[0].get("splits", []) if b.get("stats") else []
                if splits:
                    row = dict(splits[0].get("stat", {}))
                    row["Team"], row["Season"] = abbr, year
                    batting.append(row)

                p = statsapi.get("team_stats", {
                    "teamId": team_id, "season": year, "group": "pitching", "stats": "season"
                })
                splits = p.get("stats", [{}])[0].get("splits", []) if p.get("stats") else []
                if splits:
                    row = dict(splits[0].get("stat", {}))
                    row["Team"], row["Season"] = abbr, year
                    era = row.get("era")
                    row["ERA"] = pd.to_numeric(era, errors="coerce")
                    pitching.append(row)
                time.sleep(0.15)
        except Exception as exc:
            print(f"⚠️ Temporada {year}: {exc}")

    if batting:
        pd.DataFrame(batting).to_csv("data/mlb_batting.csv", index=False)
    if pitching:
        pd.DataFrame(pitching).to_csv("data/mlb_pitching.csv", index=False)
    print(f"✅ Team stats: {len(batting)} bateo / {len(pitching)} pitcheo")


def _full_rebuild_ranges():
    ranges = []
    for year in TEMPORADAS:
        for month in range(1, 13):
            last = calendar.monthrange(year, month)[1]
            ranges.append((f"{month:02d}/01/{year}", f"{month:02d}/{last:02d}/{year}"))
    return ranges


def extraer_historico_juegos():
    print("🗓️ Actualizando historial MLB...")
    path = "data/mlb_games.csv"
    existing = pd.DataFrame()
    if os.path.exists(path):
        existing = pd.read_csv(path)

    if not existing.empty and "Date" in existing.columns:
        latest = pd.to_datetime(existing.Date, errors="coerce").max()
        start = (latest - timedelta(days=3)).date() if pd.notna(latest) else date(2021, 1, 1)
        ranges = [(start.strftime("%m/%d/%Y"), date.today().strftime("%m/%d/%Y"))]
    else:
        ranges = _full_rebuild_ranges()

    new_rows = []
    ignored = {"Scheduled", "Pre-Game", "Postponed", "Cancelled", "Delayed", "Warmup", "Preview"}
    for start, end in ranges:
        for attempt in range(3):
            try:
                for game in statsapi.schedule(start_date=start, end_date=end):
                    if game.get("status", "Unknown") in ignored:
                        continue
                    if "away_score" not in game or "home_score" not in game:
                        continue
                    game_date = game.get("game_date")
                    new_rows.append({
                        "GameID": game.get("game_id"), "Date": game_date,
                        "Season": str(game_date)[:4],
                        "Away": game.get("away_name"), "Home": game.get("home_name"),
                        "Away_Score": game.get("away_score"), "Home_Score": game.get("home_score"),
                        "Innings": game.get("current_inning", 9), "Venue": game.get("venue_name", "Unknown"),
                    })
                break
            except Exception as exc:
                print(f"⚠️ {start}-{end} intento {attempt + 1}: {exc}")
                time.sleep(2)
        time.sleep(0.2)

    if not new_rows:
        print("Sin juegos nuevos")
        return
    fresh = pd.DataFrame(new_rows)
    final = pd.concat([existing, fresh], ignore_index=True) if not existing.empty else fresh
    final = final.drop_duplicates(subset=["GameID"], keep="last")
    final["Date"] = pd.to_datetime(final.Date, errors="coerce")
    final = final.sort_values("Date")
    final.to_csv(path, index=False)
    print(f"✅ Histórico guardado: {len(final)} juegos")


if __name__ == "__main__":
    extraer_estadisticas_oficiales_mlb()
    extraer_historico_juegos()
