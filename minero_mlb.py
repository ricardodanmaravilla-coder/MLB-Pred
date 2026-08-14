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

def generar_park_factors():
    import os
    import pandas as pd
    os.makedirs("data", exist_ok=True)
    datos_parks = [
        {"Team": "COL", "Estadio": "Coors Field", "Altitud": 5200, "Park_Factor": 114, "Park_Factor_HR": 115},
        {"Team": "CIN", "Estadio": "Great American Ball Park", "Altitud": 683, "Park_Factor": 107, "Park_Factor_HR": 128},
        {"Team": "BOS", "Estadio": "Fenway Park", "Altitud": 15, "Park_Factor": 106, "Park_Factor_HR": 96},
        {"Team": "LAA", "Estadio": "Angel Stadium", "Altitud": 160, "Park_Factor": 103, "Park_Factor_HR": 114},
        {"Team": "NYY", "Estadio": "Yankee Stadium", "Altitud": 54, "Park_Factor": 102, "Park_Factor_HR": 117},
        {"Team": "ATL", "Estadio": "Truist Park", "Altitud": 978, "Park_Factor": 101, "Park_Factor_HR": 105},
        {"Team": "LAD", "Estadio": "Dodger Stadium", "Altitud": 267, "Park_Factor": 100, "Park_Factor_HR": 108},
        {"Team": "HOU", "Estadio": "Minute Maid Park", "Altitud": 40, "Park_Factor": 99, "Park_Factor_HR": 97},
        {"Team": "SDP", "Estadio": "Petco Park", "Altitud": 13, "Park_Factor": 95, "Park_Factor_HR": 98},
        {"Team": "SFG", "Estadio": "Oracle Park", "Altitud": 15, "Park_Factor": 94, "Park_Factor_HR": 86},
        {"Team": "SEA", "Estadio": "T-Mobile Park", "Altitud": 10, "Park_Factor": 91, "Park_Factor_HR": 92},
        {"Team": "ARI", "Estadio": "Chase Field", "Altitud": 1086, "Park_Factor": 102, "Park_Factor_HR": 105},
        {"Team": "BAL", "Estadio": "Oriole Park at Camden Yards", "Altitud": 33, "Park_Factor": 98, "Park_Factor_HR": 105},
        {"Team": "CHC", "Estadio": "Wrigley Field", "Altitud": 610, "Park_Factor": 100, "Park_Factor_HR": 104},
        {"Team": "CLE", "Estadio": "Progressive Field", "Altitud": 580, "Park_Factor": 98, "Park_Factor_HR": 95},
        {"Team": "DET", "Estadio": "Comerica Park", "Altitud": 600, "Park_Factor": 97, "Park_Factor_HR": 91},
        {"Team": "KCR", "Estadio": "Kauffman Stadium", "Altitud": 750, "Park_Factor": 99, "Park_Factor_HR": 94},
        {"Team": "MIN", "Estadio": "Target Field", "Altitud": 840, "Park_Factor": 99, "Park_Factor_HR": 96},
        {"Team": "NYM", "Estadio": "Citi Field", "Altitud": 20, "Park_Factor": 95, "Park_Factor_HR": 94},
        {"Team": "OAK", "Estadio": "Oakland Coliseum / Sutter Health Park", "Altitud": 10, "Park_Factor": 96, "Park_Factor_HR": 90},
        {"Team": "PHI", "Estadio": "Citizens Bank Park", "Altitud": 20, "Park_Factor": 103, "Park_Factor_HR": 112},
        {"Team": "PIT", "Estadio": "PNC Park", "Altitud": 74, "Park_Factor": 97, "Park_Factor_HR": 88},
        {"Team": "STL", "Estadio": "Busch Stadium", "Altitud": 455, "Park_Factor": 96, "Park_Factor_HR": 89},
        {"Team": "TBR", "Estadio": "Tropicana Field", "Altitud": 40, "Park_Factor": 95, "Park_Factor_HR": 93},
        {"Team": "TEX", "Estadio": "Globe Life Field", "Altitud": 600, "Park_Factor": 99, "Park_Factor_HR": 102},
        {"Team": "TOR", "Estadio": "Rogers Centre", "Altitud": 250, "Park_Factor": 100, "Park_Factor_HR": 101},
        {"Team": "WSN", "Estadio": "Nationals Park", "Altitud": 20, "Park_Factor": 99, "Park_Factor_HR": 98},
        {"Team": "CHW", "Estadio": "Guaranteed Rate Field", "Altitud": 25, "Park_Factor": 98, "Park_Factor_HR": 104},
        {"Team": "MIA", "Estadio": "loanDepot park", "Altitud": 12, "Park_Factor": 93, "Park_Factor_HR": 85},
        {"Team": "MIL", "Estadio": "American Family Field", "Altitud": 632, "Park_Factor": 101, "Park_Factor_HR": 108}
    ]
    df = pd.DataFrame(datos_parks)
    df.to_csv("data/mlb_park_factors.csv", index=False)
    print("✅ Archivo mlb_park_factors.csv generado exitosamente.")

def extraer_historico_juegos():
    print("🗓️ Descargando historial de juegos (Resultados por partido)...")
    juegos_data = []
    bloques_meses = [("01/01", "03/31"), ("04/01", "05/31"), ("06/01", "07/31"), ("08/01", "09/30"), ("10/01", "12/31")]
    
    for year in TEMPORADAS:
        print(f"  -> Obteniendo calendario {year} por bloques...")
        for inicio, fin in bloques_meses:
            try:
                schedule = statsapi.schedule(start_date=f"{inicio}/{year}", end_date=f"{fin}/{year}")
                for game in schedule:
                    if game.get('status') in ['Final', 'Completed Early']:
                        juegos_data.append({
                            'GameID': game.get('game_id'), 'Date': game.get('game_date'), 'Season': year,
                            'Away': game.get('away_name'), 'Home': game.get('home_name'),
                            'Away_Score': game.get('away_score', 0), 'Home_Score': game.get('home_score', 0),
                            'Innings': game.get('current_inning', 9), 'Venue': game.get('venue_name', 'Unknown')
                        })
                time.sleep(0.3)
            except Exception as e:
                print(f"❌ Error descargando juegos de {inicio} a {fin} en {year}: {e}")
    
    df_juegos = pd.DataFrame(juegos_data)
    if not df_juegos.empty:
        df_juegos = df_juegos.drop_duplicates(subset=['GameID'])
        df_juegos.to_csv("data/mlb_games.csv", index=False)
        print(f"✅ Historial guardado: {len(df_juegos)} partidos en data/mlb_games.csv")

def descargar_abridores_individuales():
    print("DEBUG: Iniciando descargar_abridores_individuales...")
    os.makedirs("data", exist_ok=True)
    ruta_archivo = "data/mlb_pitching_individual.csv"
    
    # FORZAR CONSULTA: No dependas de la fecha de hoy si el servidor está en UTC
    # Buscaremos los pitchers de los equipos de 2026 directamente
    try:
        print("DEBUG: Consultando equipos 2026...")
        equipos = statsapi.get('teams', {'season': 2026, 'sportId': 1})['teams']
        print(f"DEBUG: Se encontraron {len(equipos)} equipos.")
        
        pitchers_data = []
        for equipo in equipos:
            tid = equipo['id']
            # Consultar roster directamente
            roster = statsapi.get('team_roster', {'teamId': tid, 'rosterType': 'active'}).get('roster', [])
            for m in roster:
                if m.get('position', {}).get('abbreviation') == 'P':
                    jugador = m.get('player', {})
                    if jugador.get('fullName'):
                        pitchers_data.append({
                            'Name': jugador.get('fullName'),
                            'Team': equipo.get('abbreviation'),
                            'ERA': 3.50, # Valor temporal, lo ajustaremos
                            'xFIP': 3.50,
                            'GS': 1
                        })
        
        df = pd.DataFrame(pitchers_data)
        print(f"DEBUG: Dataframe creado con {len(df)} filas.")
        df.to_csv(ruta_archivo, index=False)
        print(f"DEBUG: Archivo guardado en {os.path.abspath(ruta_archivo)}")
        
    except Exception as e:
        print(f"DEBUG CRÍTICO: {str(e)}")
        raise e # Esto hará que GitHub Actions marque "Error" en lugar de "Finalizado"
