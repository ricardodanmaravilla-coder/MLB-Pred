import pandas as pd
import os
import requests
from io import StringIO

def minar_stats_pitchers():
    print("⚾ [INICIO] Extrayendo estadísticas REALES desde la API interna de MLB...")
    os.makedirs("data", exist_ok=True)
    ruta_archivo = "data/mlb_pitching_individual.csv"
    
    # Endpoint oficial JSON de estadísticas de pitcheo de la temporada 2026 en MLB.com
    url = "https://statsapi.mlb.com/api/v1/stats?stats=season&season=2026&group=pitching&playerPool=ALL&limit=1000&sportId=1"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
    }
    
    try:
        print("📥 Consultando API oficial de estadísticas de jugadores...")
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            print(f"❌ Error HTTP: {response.status_code}")
            return
            
        data = response.json()
        stats_splits = data.get('stats', [])
        
        if not stats_splits:
            print("❌ La API no devolvió bloques de estadísticas.")
            return
            
        splits = stats_splits[0].get('splits', [])
        print(f"📊 Procesando registros de {len(splits)} lanzadores...")
        
        pitchers_data = []
        for split in splits:
            player = split.get('player', {})
            nombre = player.get('fullName')
            
            team_info = split.get('team', {})
            team_abbr = team_info.get('abbreviation') or team_info.get('name', 'UNK')
            
            stat = split.get('stat', {})
            era_val = stat.get('era')
            gs_val = stat.get('gamesStarted', 0)
            
            # Solo filtramos lanzadores que tengan estadísticas válidas y al menos un juego iniciado
            if nombre and era_val and era_val != '-.--':
                try:
                    era_float = float(era_val)
                except ValueError:
                    continue
                    
                pitchers_data.append({
                    'Name': nombre,
                    'Team': team_abbr,
                    'ERA': era_float,
                    'xFIP': era_float,  # Usamos ERA real como métrica base
                    'GS': int(gs_val)
                })
        
        df = pd.DataFrame(pitchers_data)
        
        if not df.empty:
            # Filtramos estrictamente a los abridores (GS > 0)
            df_abridores = df[df['GS'] > 0].copy()
            df_abridores.to_csv(ruta_archivo, index=False)
            print(f"✅ [ÉXITO] Archivo guardado con {len(df_abridores)} abridores reales.")
        else:
            print("⚠️ No se encontraron registros de lanzadores válidos en el JSON.")
            
    except Exception as e:
        print(f"❌ Error crítico en la descarga por API: {e}")

if __name__ == "__main__":
    minar_stats_pitchers()
