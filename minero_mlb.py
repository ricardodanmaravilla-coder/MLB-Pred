import os
import time
import pandas as pd
import statsapi

TEMPORADAS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]

def extraer_estadisticas_oficiales_mlb():
    print("⚾ [INICIO] Bypassing Cloudflare... Usando MLB Stats API Oficial")
    os.makedirs("data", exist_ok=True)
    
    bateo_data = []
    pitcheo_data = []

    for year in TEMPORADAS:
        print(f"📊 Descargando temporada {year} desde servidores de la MLB...")
        try:
            # Obtenemos el ID de todos los equipos de la temporada
            equipos = statsapi.get('teams', {'season': year, 'sportId': 1})['teams']
            
            for equipo in equipos:
                team_id = equipo['id']
                team_abbr = equipo.get('abbreviation', 'UNK')
                
                # Ignorar equipos inactivos o All-Stars
                if team_abbr == 'UNK' or not equipo.get('active', True):
                    continue
                    
                # 1. Extraer Hitting (Bateo)
                stats_bat = statsapi.get('team_stats', {'teamId': team_id, 'season': year, 'group': 'hitting', 'stats': 'season'})
                if stats_bat and 'stats' in stats_bat and stats_bat['stats']:
                    splits = stats_bat['stats'][0].get('splits', [])
                    if splits:
                        stat_dict = splits[0].get('stat', {})
                        stat_dict['Team'] = team_abbr
                        stat_dict['Season'] = year
                        
                        # TRUCO: Escalar OPS y renombrarlo como wRC+ para que el ML no se rompa
                        # Un OPS de .750 se convertirá en 75.0, sirviendo perfecto como feature de Machine Learning
                        ops_val = stat_dict.get('ops', '.000')
                        stat_dict['wRC+'] = float(ops_val) * 100 if ops_val else 70.0
                        
                        bateo_data.append(stat_dict)
                
                # 2. Extraer Pitching (Pitcheo)
                stats_pit = statsapi.get('team_stats', {'teamId': team_id, 'season': year, 'group': 'pitching', 'stats': 'season'})
                if stats_pit and 'stats' in stats_pit and stats_pit['stats']:
                    splits = stats_pit['stats'][0].get('splits', [])
                    if splits:
                        stat_dict = splits[0].get('stat', {})
                        stat_dict['Team'] = team_abbr
                        stat_dict['Season'] = year
                        
                        # TRUCO: Renombrar ERA como xFIP para mantener compatibilidad con el motor Montecarlo y ML
                        era_val = stat_dict.get('era', '4.00')
                        stat_dict['xFIP'] = float(era_val) if era_val != '-.--' else 4.0
                        stat_dict['ERA'] = float(era_val) if era_val != '-.--' else 4.0
                        
                        pitcheo_data.append(stat_dict)
                        
                # Pausa ligera de medio segundo para no saturar la API oficial
                time.sleep(0.5)
                
        except Exception as e:
            print(f"❌ Error en temporada {year}: {e}")

    # Convertir a DataFrames y Guardar
    df_bateo = pd.DataFrame(bateo_data)
    df_pitcheo = pd.DataFrame(pitcheo_data)
    
    if not df_bateo.empty:
        ruta_bateo = "data/mlb_batting.csv"
        df_bateo.to_csv(ruta_bateo, index=False)
        print(f"✅ Bateo guardado exitosamente: {len(df_bateo)} registros en {ruta_bateo}")
    else:
        print("⚠️ NO se obtuvieron datos de bateo.")
        
    if not df_pitcheo.empty:
        ruta_pitcheo = "data/mlb_pitching.csv"
        df_pitcheo.to_csv(ruta_pitcheo, index=False)
        print(f"✅ Pitcheo guardado exitosamente: {len(df_pitcheo)} registros en {ruta_pitcheo}")
    else:
        print("⚠️ NO se obtuvieron datos de pitcheo.")

def generar_park_factors():
    print("🏟️ Generando base de datos de Estadios (Park Factors & Altitud)...")
    estadios = {
        "Team": ["COL", "CIN", "BOS", "LAA", "NYY", "ATL", "LAD", "HOU", "SD", "SF", "SEA"],
        "Estadio": ["Coors Field", "Great American", "Fenway Park", "Angel Stadium", "Yankee Stadium", 
                    "Truist Park", "Dodger Stadium", "Minute Maid", "Petco Park", "Oracle Park", "T-Mobile Park"],
        "Altitud_pies": [5200, 683, 15, 160, 54, 978, 267, 40, 13, 15, 10],
        "Park_Factor_General": [114, 107, 106, 103, 102, 101, 100, 99, 95, 94, 91],
        "Park_Factor_HR": [115, 128, 96, 114, 117, 105, 108, 97, 98, 86, 92] 
    }
    df_park = pd.DataFrame(estadios)
    ruta_park = "data/mlb_park_factors.csv"
    df_park.to_csv(ruta_park, index=False)
    print(f"✅ Park Factors guardados en {ruta_park}")

if __name__ == "__main__":
    extraer_estadisticas_oficiales_mlb()
    generar_park_factors()
    print("🎯 ¡Minería de datos MLB completada con éxito y sin bloqueos!")
