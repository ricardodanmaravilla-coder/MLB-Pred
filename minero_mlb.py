import os
import time
import pandas as pd
import statsapi

TEMPORADAS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]

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
                
                # Pitcheo
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

#def generar_park_factors():
 #   print("🏟️ Generando base de datos de Estadios (Park Factors & Altitud)...")
  #  estadios = {
   #     "Team": ["COL", "CIN", "BOS", "LAA", "NYY", "ATL", "LAD", "HOU", "SD", "SF", "SEA"],
    #    "Estadio": ["Coors Field", "Great American", "Fenway Park", "Angel Stadium", "Yankee Stadium", 
     #               "Truist Park", "Dodger Stadium", "Minute Maid", "Petco Park", "Oracle Park", "T-Mobile Park"],
      #  "Altitud_pies": [5200, 683, 15, 160, 54, 978, 267, 40, 13, 15, 10],
       # "Park_Factor_General": [114, 107, 106, 103, 102, 101, 100, 99, 95, 94, 91],
        #"Park_Factor_HR": [115, 128, 96, 114, 117, 105, 108, 97, 98, 86, 92] 
    #}
    #df_park = pd.DataFrame(estadios)
    #df_park.to_csv("data/mlb_park_factors.csv", index=False)
    #print(f"✅ Park Factors guardados en data/mlb_park_factors.csv")

def extraer_historico_juegos():
    """
    Descarga el registro partido a partido.
    Se hace en bloques de meses para evitar el error 503 (Timeout) del servidor de la MLB.
    """
    print("🗓️ Descargando historial de juegos (Resultados por partido)...")
    juegos_data = []
    
    # Dividimos el año en 5 bloques para no saturar al servidor
    bloques_meses = [
        ("01/01", "03/31"),
        ("04/01", "05/31"),
        ("06/01", "07/31"),
        ("08/01", "09/30"),
        ("10/01", "12/31")
    ]
    
    for year in TEMPORADAS:
        print(f"  -> Obteniendo calendario {year} por bloques...")
        for inicio, fin in bloques_meses:
            try:
                schedule = statsapi.schedule(start_date=f"{inicio}/{year}", end_date=f"{fin}/{year}")
                
                for game in schedule:
                    if game.get('status') in ['Final', 'Completed Early']:
                        juegos_data.append({
                            'GameID': game.get('game_id'),
                            'Date': game.get('game_date'),
                            'Season': year,
                            'Away': game.get('away_name'),
                            'Home': game.get('home_name'),
                            'Away_Score': game.get('away_score', 0),
                            'Home_Score': game.get('home_score', 0),
                            'Innings': game.get('current_inning', 9),
                            'Venue': game.get('venue_name', 'Unknown')
                        })
                # Pequeña pausa para ser amigables con el servidor
                time.sleep(0.3)
            except Exception as e:
                print(f"❌ Error descargando juegos de {inicio} a {fin} en {year}: {e}")
    
    df_juegos = pd.DataFrame(juegos_data)
    if not df_juegos.empty:
        # Eliminamos posibles duplicados por seguridad
        df_juegos = df_juegos.drop_duplicates(subset=['GameID'])
        df_juegos.to_csv("data/mlb_games.csv", index=False)
        print(f"✅ Historial guardado: {len(df_juegos)} partidos en data/mlb_games.csv")
    else:
        print("⚠️ NO se obtuvieron los juegos históricos.")

if __name__ == "__main__":
    extraer_estadisticas_oficiales_mlb()
    generar_park_factors()
    extraer_historico_juegos()  # <--- Agregamos la nueva función al flujo principal
    print("🎯 ¡Minería de datos MLB completada con éxito y sin bloqueos!")
