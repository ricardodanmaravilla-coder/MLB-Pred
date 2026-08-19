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
                        
                time.sleep(0.3)
        except Exception as e:
            print(f"❌ Error en temporada {year}: {e}")

    df_bateo = pd.DataFrame(bateo_data)
    df_pitcheo = pd.DataFrame(pitcheo_data)
    
    if not df_bateo.empty:
        df_bateo.to_csv("data/mlb_batting.csv", index=False)
        print("✅ Bateo guardado exitosamente.")
        
    if not df_pitcheo.empty:
        df_pitcheo.to_csv("data/mlb_pitching.csv", index=False)
        print("✅ Pitcheo guardado exitosamente.")

def extraer_historico_juegos():
    print("\n🗓️ Actualizando historial de juegos de forma híbrida...")
    archivo_csv = "data/mlb_games.csv"
    filas_antes = 0
    df_existente = pd.DataFrame()
    tramos_descarga = []
    
    # 1. Decidir la estrategia de descarga (Rápida vs Reconstrucción Total)
    if os.path.exists(archivo_csv):
        df_existente = pd.read_csv(archivo_csv)
        filas_antes = len(df_existente)
        ultima_fecha = pd.to_datetime(df_existente['Date']).max()
        print(f"✅ Archivo encontrado con {filas_antes} partidos. Último: {ultima_fecha.strftime('%Y-%m-%d')}")
        
        # Estrategia Rápida: Solo pedimos los últimos días en un solo tramo
        inicio = ultima_fecha - timedelta(days=3)
        fin = date.today()
        tramos_descarga.append((inicio.strftime('%m/%d/%Y'), fin.strftime('%m/%d/%Y')))
    else:
        print("⚠️ No se encontró archivo. Construyendo desde 2021 por bloques para no saturar la API...")
        bloques = [("01/01", "03/31"), ("04/01", "05/31"), ("06/01", "07/31"), ("08/01", "09/30"), ("10/01", "12/31")]
        
        # Estrategia Total: Partimos los 6 años en pedazos de 2-3 meses
        for year in TEMPORADAS:
            for b_inicio, b_fin in bloques:
                tramos_descarga.append((f"{b_inicio}/{year}", f"{b_fin}/{year}"))

    nuevos_juegos = []
    estados_ignorados = ['Scheduled', 'Pre-Game', 'Postponed', 'Cancelled', 'Delayed', 'Warmup', 'Preview']
    
    # 2. Ejecutar descargas por tramos para evitar bloqueos
    for inc_str, fin_str in tramos_descarga:
        print(f"🔍 Buscando tramo: {inc_str} a {fin_str}...")
        try:
            schedule = statsapi.schedule(start_date=inc_str, end_date=fin_str)
            for game in schedule:
                status = game.get('status', 'Unknown')
                if status not in estados_ignorados and 'away_score' in game and 'home_score' in game:
                    nuevos_juegos.append({
                        'GameID': game.get('game_id'), 'Date': game.get('game_date'), 'Season': str(game.get('game_date'))[:4],
                        'Away': game.get('away_name'), 'Home': game.get('home_name'),
                        'Away_Score': game.get('away_score', 0), 'Home_Score': game.get('home_score', 0),
                        'Innings': game.get('current_inning', 9), 'Venue': game.get('venue_name', 'Unknown')
                    })
            time.sleep(0.3)
        except Exception as e:
            print(f"❌ Error descargando tramo {inc_str}-{fin_str}: {e}")

    # 3. Consolidación y Guardado
    if nuevos_juegos:
        df_nuevos = pd.DataFrame(nuevos_juegos)
        print(f"📥 La API entregó un total de {len(df_nuevos)} juegos en esta sesión.")
        
        if not df_existente.empty:
            df_final = pd.concat([df_existente, df_nuevos])
        else:
            df_final = df_nuevos
            
        df_final = df_final.drop_duplicates(subset=['GameID'], keep='last')
        filas_despues = len(df_final)
        
        if filas_despues > filas_antes:
            print(f"🚀 ¡Se han añadido {filas_despues - filas_antes} partidos nuevos al archivo!")
        else:
            print("🔄 Se actualizaron scores recientes, sin añadir filas nuevas.")
            
        try:
            df_final.to_csv(archivo_csv, index=False)
            print(f"✅ ¡ÉXITO! Archivo {archivo_csv} guardado con {len(df_final)} partidos totales.")
        except PermissionError:
            print("❌ ERROR: El archivo mlb_games.csv está bloqueado por otro programa.")
    else:
        print("🤷‍♂️ No hay partidos nuevos extraídos de la API.")

if __name__ == "__main__":
    print("--- INICIANDO SCRIPT DE MINERÍA ---")
    print(f"Directorio actual: {os.getcwd()}")
    os.makedirs("data", exist_ok=True)
    
    extraer_estadisticas_oficiales_mlb()
    extraer_historico_juegos()
    
    print("🎯 ¡Minería completada con éxito!")
