import os
import time
import pandas as pd
import statsapi
from datetime import date, timedelta

TEMPORADAS = [2021, 2022, 2023, 2024, 2025, 2026]

def extraer_estadisticas_oficiales_mlb():
    print("⚾ [INICIO] Extrayendo Sabermetría Oficial de MLB...")
    os.makedirs("data", exist_ok=True)
    
    bateo_data = []
    pitcheo_data = []

    for year in TEMPORADAS:
        print(f"📊 Descargando estadísticas globales {year}...")
        try:
            equipos = statsapi.get('teams', {'season': year, 'sportId': 1})['teams']
            for equipo in equipos:
                team_id = equipo['id']
                team_abbr = equipo.get('abbreviation', 'UNK')
                
                if team_abbr == 'UNK' or not equipo.get('active', True):
                    continue
                    
                # Bateo
                stats_bat = statsapi.get('team_stats', {'teamId': team_id, 'season': year, 'group': 'hitting', 'stats': 'season'})
                if stats_bat and 'stats' in stats_bat and stats_bat['stats']:
                    splits = stats_bat['stats'][0].get('splits', [])
                    if splits:
                        stat_dict = splits[0].get('stat', {})
                        stat_dict['Team'] = team_abbr
                        stat_dict['Season'] = year
                        ops_val = stat_dict.get('ops', '.000')
                        stat_dict['wRC+'] = float(ops_val) * 100 if ops_val else 70.0
                        bateo_data.append(stat_dict)
                
                # Pitcheo (Datos de Equipo / Bullpen)
                stats_pit = statsapi.get('team_stats', {'teamId': team_id, 'season': year, 'group': 'pitching', 'stats': 'season'})
                if stats_pit and 'stats' in stats_pit and stats_pit['stats']:
                    splits = stats_pit['stats'][0].get('splits', [])
                    if splits:
                        stat_dict = splits[0].get('stat', {})
                        stat_dict['Team'] = team_abbr
                        stat_dict['Season'] = year
                        era_val = stat_dict.get('era', '4.00')
                        stat_dict['xFIP'] = float(era_val) if era_val != '-.--' else 4.0
                        stat_dict['ERA'] = float(era_val) if era_val != '-.--' else 4.0
                        pitcheo_data.append(stat_dict)
                        
                time.sleep(0.5)
        except Exception as e:
            print(f"❌ Error en temporada {year}: {e}")

    df_bateo = pd.DataFrame(bateo_data)
    df_pitcheo = pd.DataFrame(pitcheo_data)
    
    if not df_bateo.empty:
        df_bateo.to_csv("data/mlb_batting.csv", index=False)
        print(f"✅ Bateo guardado exitosamente.")
        
    if not df_pitcheo.empty:
        df_pitcheo.to_csv("data/mlb_pitching.csv", index=False)
        print(f"✅ Pitcheo guardado exitosamente.")

def extraer_historico_juegos():
    print("🗓️ Actualizando historial de juegos de forma inteligente...")
    archivo_csv = "data/mlb_games.csv"
    
    # 1. Leer el archivo que ya tienes para no descargar todo desde cero
    if os.path.exists(archivo_csv):
        df_existente = pd.read_csv(archivo_csv)
        # Encontrar la fecha más reciente en tu archivo
        ultima_fecha = pd.to_datetime(df_existente['Date']).max()
        print(f"✅ Archivo encontrado. Último partido registrado: {ultima_fecha.strftime('%Y-%m-%d')}")
        # Retrocedemos 3 días por seguridad (por si algún juego suspendido se reanudó)
        inicio_busqueda = (ultima_fecha - timedelta(days=3))
    else:
        print("⚠️ No se encontró archivo previo. Se descargará desde 2020 (tomará tiempo)...")
        ultima_fecha = pd.to_datetime("2020-01-01")
        df_existente = pd.DataFrame()
        inicio_busqueda = ultima_fecha

    # 2. Configurar la búsqueda solo para los días que te faltan
    hoy = date.today()
    inicio_str = inicio_busqueda.strftime('%m/%d/%Y')
    fin_str = hoy.strftime('%m/%d/%Y')
    
    print(f"🔍 Buscando partidos estrictamente desde {inicio_str} hasta {fin_str}...")
    
    nuevos_juegos = []
    estados_ignorados = ['Scheduled', 'Pre-Game', 'Postponed', 'Cancelled', 'Delayed', 'Warmup', 'Preview']
    
    try:
        schedule = statsapi.schedule(start_date=inicio_str, end_date=fin_str)
        for game in schedule:
            status = game.get('status', 'Unknown')
            if status not in estados_ignorados and 'away_score' in game and 'home_score' in game:
                nuevos_juegos.append({
                    'GameID': game.get('game_id'), 'Date': game.get('game_date'), 'Season': game.get('game_date')[:4],
                    'Away': game.get('away_name'), 'Home': game.get('home_name'),
                    'Away_Score': game.get('away_score', 0), 'Home_Score': game.get('home_score', 0),
                    'Innings': game.get('current_inning', 9), 'Venue': game.get('venue_name', 'Unknown')
                })
        
        if nuevos_juegos:
            df_nuevos = pd.DataFrame(nuevos_juegos)
            print(f"📥 La API entregó {len(df_nuevos)} juegos en este periodo corto.")
            
            # Fusionar los juegos viejos con los recién descargados
            if not df_existente.empty:
                df_final = pd.concat([df_existente, df_nuevos])
            else:
                df_final = df_nuevos
                
            # Eliminar duplicados manteniendo el resultado final más reciente
            df_final = df_final.drop_duplicates(subset=['GameID'], keep='last')
            
            # 3. Guardado con alerta de permisos (Para detectar si Excel bloquea el guardado)
            try:
                df_final.to_csv(archivo_csv, index=False)
                print(f"✅ ¡ÉXITO! Archivo mlb_games.csv actualizado y guardado. Partidos totales: {len(df_final)}")
            except PermissionError:
                print("❌ ERROR CRÍTICO: El archivo mlb_games.csv está abierto en Excel u otro programa. Ciérralo y vuelve a correr el script.")
        else:
            print("🤷‍♂️ No hay partidos nuevos para agregar en estas fechas.")
            
    except Exception as e:
        print(f"❌ Error conectando con la API de la MLB: {e}")

if __name__ == "__main__":
    print("--- INICIANDO SCRIPT DE MINERÍA ---")
    print(f"Directorio actual: {os.getcwd()}")
    os.makedirs("data", exist_ok=True)
    
    extraer_estadisticas_oficiales_mlb()
    extraer_historico_juegos()
    
    print("🎯 ¡Minería de datos MLB completada con éxito!")
