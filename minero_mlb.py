import calendar
import os
import time
from datetime import date, timedelta

import pandas as pd
import statsapi

from modules.advanced_stats import enrich_team_frames
from modules.team_utils import normalize_team

TEMPORADAS = [2021, 2022, 2023, 2024, 2025, 2026]


def _safe_float(value, default=None):
    try:
        if value in (None, "", "-.--"):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def extraer_estadisticas_oficiales_mlb():
    print("⚾ [INICIO] Extrayendo estadísticas oficiales de MLB StatsAPI...")
    os.makedirs("data", exist_ok=True)
    bateo_data, pitcheo_data = [], []

    for year in TEMPORADAS:
        print(f"📊 Descargando estadísticas globales {year}...")
        try:
            equipos = statsapi.get("teams", {"season": year, "sportId": 1})["teams"]
            for equipo in equipos:
                team_id = equipo["id"]
                team_abbr = normalize_team(equipo.get("abbreviation", "UNK"))
                if not team_abbr or team_abbr == "UNK" or not equipo.get("active", True):
                    continue

                stats_bat = statsapi.get("team_stats", {"teamId": team_id, "season": year, "group": "hitting", "stats": "season"})
                if stats_bat and stats_bat.get("stats"):
                    splits = stats_bat["stats"][0].get("splits", [])
                    if splits:
                        stat = dict(splits[0].get("stat", {}))
                        stat["Team"] = team_abbr
                        stat["Season"] = year
                        ops = _safe_float(stat.get("ops"))
                        stat["OPS"] = ops
                        stat["OPS_Index"] = None if ops is None else ops * 100.0
                        stat["wRC+"] = stat["OPS_Index"]
                        stat["wRC+_Source"] = "LEGACY_OPS_X100_NOT_REAL_WRCPLUS"
                        bateo_data.append(stat)

                stats_pit = statsapi.get("team_stats", {"teamId": team_id, "season": year, "group": "pitching", "stats": "season"})
                if stats_pit and stats_pit.get("stats"):
                    splits = stats_pit["stats"][0].get("splits", [])
                    if splits:
                        stat = dict(splits[0].get("stat", {}))
                        stat["Team"] = team_abbr
                        stat["Season"] = year
                        era = _safe_float(stat.get("era"))
                        stat["ERA"] = era
                        stat["xFIP"] = era
                        stat["xFIP_Source"] = "LEGACY_ERA_PROXY_NOT_REAL_XFIP"
                        pitcheo_data.append(stat)

                time.sleep(0.20)
        except Exception as exc:
            print(f"❌ Error en temporada {year}: {exc}")

    df_bateo = pd.DataFrame(bateo_data)
    df_pitcheo = pd.DataFrame(pitcheo_data)
    if df_bateo.empty or df_pitcheo.empty:
        print('❌ StatsAPI no produjo bases de equipo suficientes.')
        return df_bateo, df_pitcheo

    df_bateo, df_pitcheo, bullpen_fg = enrich_team_frames(df_bateo, df_pitcheo, TEMPORADAS)
    df_bateo.to_csv("data/mlb_batting.csv", index=False)
    df_pitcheo.to_csv("data/mlb_pitching.csv", index=False)
    print("✅ Bateo guardado; wRC+ real se usa cuando FanGraphs respondió.")
    print("✅ Pitcheo guardado; FIP/xFIP reales se usan cuando FanGraphs respondió.")

    if bullpen_fg is not None and not bullpen_fg.empty:
        current = bullpen_fg[pd.to_numeric(bullpen_fg['Season'], errors='coerce') == max(TEMPORADAS)].copy()
        if current['Team'].nunique() >= 25:
            current.to_csv('data/mlb_bullpen_fangraphs.csv', index=False)
            print(f"✅ Bullpen FanGraphs real guardado: {current['Team'].nunique()} equipos.")
    return df_bateo, df_pitcheo


def _monthly_ranges(year):
    ranges = []
    for month in range(1, 10):
        last = calendar.monthrange(year, month)[1]
        ranges.append((f"{month:02d}/01/{year}", f"{month:02d}/{last:02d}/{year}"))
    ranges.append((f"10/01/{year}", f"12/31/{year}"))
    return ranges


def extraer_historico_juegos():
    print("\n🗓️ Actualizando historial de juegos...")
    archivo_csv = "data/mlb_games.csv"
    filas_antes = 0
    df_existente = pd.DataFrame()
    tramos_descarga = []

    if os.path.exists(archivo_csv):
        df_existente = pd.read_csv(archivo_csv)
        filas_antes = len(df_existente)
        ultima_fecha = pd.to_datetime(df_existente["Date"], errors="coerce").max()
        if pd.notna(ultima_fecha):
            inicio = ultima_fecha - timedelta(days=3)
            fin = date.today()
            tramos_descarga.append((inicio.strftime("%m/%d/%Y"), fin.strftime("%m/%d/%Y")))
            print(f"✅ Archivo encontrado con {filas_antes} partidos. Último: {ultima_fecha.strftime('%Y-%m-%d')}")
    if not tramos_descarga:
        print("⚠️ Reconstrucción mensual desde 2021.")
        for year in TEMPORADAS:
            tramos_descarga.extend(_monthly_ranges(year))

    nuevos_juegos = []
    estados_ignorados = {"Scheduled", "Pre-Game", "Postponed", "Cancelled", "Delayed", "Warmup", "Preview"}
    for inc_str, fin_str in tramos_descarga:
        print(f"🔍 {inc_str} a {fin_str}...")
        for intento in range(3):
            try:
                schedule = statsapi.schedule(start_date=inc_str, end_date=fin_str)
                for game in schedule:
                    status = game.get("status", "Unknown")
                    if status in estados_ignorados or "away_score" not in game or "home_score" not in game:
                        continue
                    game_date = game.get("game_date")
                    nuevos_juegos.append({
                        "GameID": game.get("game_id"),
                        "Date": game_date,
                        "Season": str(game_date)[:4],
                        "GameType": game.get("game_type") or game.get("gameType"),
                        "Away": game.get("away_name"),
                        "Home": game.get("home_name"),
                        "Away_Score": game.get("away_score", 0),
                        "Home_Score": game.get("home_score", 0),
                        "Away_Starter": game.get("away_probable_pitcher") or game.get("away_pitcher"),
                        "Home_Starter": game.get("home_probable_pitcher") or game.get("home_pitcher"),
                        "Innings": game.get("current_inning", 9),
                        "Venue": game.get("venue_name", "Unknown"),
                    })
                time.sleep(0.35)
                break
            except Exception as exc:
                print(f"⚠️ Intento {intento + 1}: {exc}")
                time.sleep(2)

    if not nuevos_juegos:
        print("🤷‍♂️ No hay partidos nuevos.")
        return

    df_nuevos = pd.DataFrame(nuevos_juegos)
    df_final = pd.concat([df_existente, df_nuevos], ignore_index=True) if not df_existente.empty else df_nuevos
    if 'GameID' in df_final.columns:
        with_id = df_final[df_final['GameID'].notna()].drop_duplicates(subset=["GameID"], keep="last")
        without_id = df_final[df_final['GameID'].isna()].drop_duplicates(subset=['Date','Away','Home'], keep='last')
        df_final = pd.concat([with_id, without_id], ignore_index=True)
    else:
        df_final = df_final.drop_duplicates(subset=['Date','Away','Home'], keep='last')
    df_final.to_csv(archivo_csv, index=False)
    print(f"✅ {archivo_csv}: {len(df_final)} partidos ({len(df_final) - filas_antes:+d}).")


if __name__ == "__main__":
    print("--- INICIANDO SCRIPT DE MINERÍA V6 ---")
    os.makedirs("data", exist_ok=True)
    extraer_estadisticas_oficiales_mlb()
    extraer_historico_juegos()
    print("🎯 Minería completada.")
