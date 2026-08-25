import calendar
import os
import time
from datetime import date, timedelta

import pandas as pd
import statsapi

from modules.fangraphs_mlb import fetch_team_fangraphs
from modules.team_utils import normalize_team

TEMPORADAS = [2021, 2022, 2023, 2024, 2025, 2026]


def _safe_float(value, default=None):
    try:
        if value in (None, "", "-.--"):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _statsapi_team_fallback():
    bateo_data, pitcheo_data = [], []
    for year in TEMPORADAS:
        print(f"📊 MLB StatsAPI fallback {year}...")
        try:
            equipos = statsapi.get("teams", {"season": year, "sportId": 1})["teams"]
            for equipo in equipos:
                team_id = equipo["id"]
                team_abbr = normalize_team(equipo.get("abbreviation", "UNK"))
                if not team_abbr or not equipo.get("active", True):
                    continue
                stats_bat = statsapi.get("team_stats", {"teamId": team_id, "season": year, "group": "hitting", "stats": "season"})
                if stats_bat and stats_bat.get("stats"):
                    splits = stats_bat["stats"][0].get("splits", [])
                    if splits:
                        stat = dict(splits[0].get("stat", {})); stat["Team"] = team_abbr; stat["Season"] = year
                        ops = _safe_float(stat.get("ops")); stat["OPS"] = ops; stat["OPS_Index"] = None if ops is None else ops * 100.0
                        stat["wRC+"] = stat["OPS_Index"]; stat["wRC+_Source"] = "LEGACY_OPS_X100_NOT_REAL_WRCPLUS"
                        stat["DataSource"] = "MLB_STATSAPI_FALLBACK"; bateo_data.append(stat)
                stats_pit = statsapi.get("team_stats", {"teamId": team_id, "season": year, "group": "pitching", "stats": "season"})
                if stats_pit and stats_pit.get("stats"):
                    splits = stats_pit["stats"][0].get("splits", [])
                    if splits:
                        stat = dict(splits[0].get("stat", {})); stat["Team"] = team_abbr; stat["Season"] = year
                        era = _safe_float(stat.get("era")); stat["ERA"] = era; stat["xFIP"] = era
                        stat["xFIP_Source"] = "LEGACY_ERA_PROXY_NOT_REAL_XFIP"; stat["DataSource"] = "MLB_STATSAPI_FALLBACK"
                        pitcheo_data.append(stat)
                time.sleep(0.12)
        except Exception as exc:
            print(f"❌ Error StatsAPI {year}: {exc}")
    return pd.DataFrame(bateo_data), pd.DataFrame(pitcheo_data)


def extraer_estadisticas_oficiales_mlb():
    """Use real FanGraphs advanced team metrics; fall back safely to MLB StatsAPI."""
    print("⚾ [INICIO] Extrayendo métricas avanzadas MLB/FanGraphs...")
    os.makedirs("data", exist_ok=True)
    fg_bat, fg_pit, fg_rel = fetch_team_fangraphs(min(TEMPORADAS), max(TEMPORADAS))
    fallback_bat, fallback_pit = _statsapi_team_fallback()

    if not fg_bat.empty and {'Team', 'Season', 'wRC+'}.issubset(fg_bat.columns):
        bat = fg_bat.copy()
        bat['Team'] = bat['Team'].map(normalize_team)
        bat['wRC+_Source'] = 'FANGRAPHS_REAL_WRCPLUS'
        if 'OPS' in bat.columns:
            bat['OPS_Index'] = pd.to_numeric(bat['OPS'], errors='coerce') * 100.0
        bat.to_csv("data/mlb_batting.csv", index=False)
        print(f"✅ Bateo FanGraphs real: {len(bat)} filas; wRC+/wOBA/ISO/K%/BB% cuando disponibles.")
    elif not fallback_bat.empty:
        fallback_bat.to_csv("data/mlb_batting.csv", index=False)
        print("⚠️ FanGraphs no disponible; bateo guardado con fallback StatsAPI.")

    if not fg_pit.empty and {'Team', 'Season', 'xFIP'}.issubset(fg_pit.columns):
        pit = fg_pit.copy(); pit['Team'] = pit['Team'].map(normalize_team); pit['xFIP_Source'] = 'FANGRAPHS_REAL_XFIP'
        pit.to_csv("data/mlb_pitching.csv", index=False)
        print(f"✅ Pitcheo FanGraphs real: {len(pit)} filas; xFIP/FIP/K-BB%/WHIP/HR9 cuando disponibles.")
    elif not fallback_pit.empty:
        fallback_pit.to_csv("data/mlb_pitching.csv", index=False)
        print("⚠️ FanGraphs no disponible; pitcheo guardado con fallback ERA.")

    if not fg_rel.empty and {'Team', 'Season'}.issubset(fg_rel.columns):
        rel = fg_rel.copy(); rel['Team'] = rel['Team'].map(normalize_team); rel['Source'] = 'FANGRAPHS_REAL_RELIEVERS'
        if 'xFIP' in rel.columns:
            rel['ERA_Estimator'] = pd.to_numeric(rel['xFIP'], errors='coerce').combine_first(pd.to_numeric(rel.get('FIP'), errors='coerce')).combine_first(pd.to_numeric(rel.get('ERA'), errors='coerce'))
        rel.to_csv('data/mlb_bullpen_advanced.csv', index=False)
        print(f"✅ Bullpen FanGraphs avanzado: {len(rel)} filas.")


def _monthly_ranges(year):
    ranges = []
    for month in range(1, 10):
        last = calendar.monthrange(year, month)[1]
        ranges.append((f"{month:02d}/01/{year}", f"{month:02d}/{last:02d}/{year}"))
    ranges.append((f"10/01/{year}", f"12/31/{year}"))
    return ranges


def extraer_historico_juegos():
    print("\n🗓️ Actualizando historial de juegos...")
    archivo_csv = "data/mlb_games.csv"; filas_antes = 0; df_existente = pd.DataFrame(); tramos_descarga = []
    if os.path.exists(archivo_csv):
        df_existente = pd.read_csv(archivo_csv); filas_antes = len(df_existente)
        ultima_fecha = pd.to_datetime(df_existente["Date"], errors="coerce").max()
        if pd.notna(ultima_fecha):
            inicio = ultima_fecha - timedelta(days=3); fin = date.today()
            tramos_descarga.append((inicio.strftime("%m/%d/%Y"), fin.strftime("%m/%d/%Y")))
    if not tramos_descarga:
        for year in TEMPORADAS: tramos_descarga.extend(_monthly_ranges(year))

    nuevos_juegos = []; estados_ignorados = {"Scheduled", "Pre-Game", "Postponed", "Cancelled", "Delayed", "Warmup", "Preview"}
    for inc_str, fin_str in tramos_descarga:
        for intento in range(3):
            try:
                schedule = statsapi.schedule(start_date=inc_str, end_date=fin_str)
                for game in schedule:
                    status = game.get("status", "Unknown")
                    if status in estados_ignorados or "away_score" not in game or "home_score" not in game: continue
                    game_date = game.get("game_date")
                    nuevos_juegos.append({
                        "GameID": game.get("game_id"), "Date": game_date, "Season": str(game_date)[:4],
                        "GameType": game.get("game_type") or game.get("gameType"), "Away": game.get("away_name"), "Home": game.get("home_name"),
                        "Away_Score": game.get("away_score", 0), "Home_Score": game.get("home_score", 0),
                        "Away_Starter": game.get("away_probable_pitcher") or game.get("away_pitcher"),
                        "Home_Starter": game.get("home_probable_pitcher") or game.get("home_pitcher"),
                        "Innings": game.get("current_inning", 9), "Venue": game.get("venue_name", "Unknown"),
                    })
                time.sleep(0.3); break
            except Exception as exc:
                print(f"⚠️ Intento {intento + 1}: {exc}"); time.sleep(2)
    if not nuevos_juegos: return
    df_nuevos = pd.DataFrame(nuevos_juegos)
    df_final = pd.concat([df_existente, df_nuevos], ignore_index=True) if not df_existente.empty else df_nuevos
    if 'GameID' in df_final.columns: df_final = df_final.drop_duplicates(subset=["GameID"], keep="last")
    df_final.to_csv(archivo_csv, index=False)
    print(f"✅ {archivo_csv}: {len(df_final)} partidos ({len(df_final) - filas_antes:+d}).")


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    extraer_estadisticas_oficiales_mlb(); extraer_historico_juegos(); print("🎯 Minería completada.")
