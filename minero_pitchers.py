import pandas as pd
import os
import requests
from io import StringIO

def minar_stats_pitchers():
    print("⚾ [INICIO] Extrayendo estadísticas REALES de lanzadores (MLB Data Feed)...")
    os.makedirs("data", exist_ok=True)
    
    # URL oficial de la MLB para estadísticas de pitcheo (esto contiene datos reales)
    # Usamos la temporada 2026 actual
    url = "https://www.mlb.com/stats/pitching?playerPool=ALL&season=2026"
    
    try:
        # Nota: Como es una página web con tabla, usaremos pandas para leer la tabla directamente
        # Esto es más efectivo que APIs que bloquean peticiones
        print("📥 Conectando con mlb.com...")
        dfs = pd.read_html("https://www.mlb.com/stats/pitching?playerPool=ALL&season=2026")
        
        if not dfs:
            print("❌ No se pudieron leer las tablas.")
            return

        # La tabla suele ser la primera que aparece
        df = dfs[0]
        
        # Limpieza: Asegurarnos de que las columnas coincidan con lo que tu app necesita
        # MLB.com suele poner el nombre y equipo juntos, vamos a separarlos si es necesario
        df = df[['Player', 'Team', 'ERA', 'IP', 'GS']]
        df.rename(columns={'Player': 'Name', 'IP': 'Innings'}, inplace=True)
        
        # Filtrar solo a los que tienen juegos iniciados (GS > 0) para tener abridores reales
        df_abridores = df[df['GS'] > 0].copy()
        
        # Añadir columna xFIP como espejo de ERA para tu modelo
        df_abridores['xFIP'] = df_abridores['ERA']
        
        # Guardar
        df_abridores.to_csv("data/mlb_pitching_individual.csv", index=False)
        print(f"✅ [ÉXITO] Archivo guardado con {len(df_abridores)} lanzadores reales.")
        
    except Exception as e:
        print(f"❌ Error en la descarga: {e}")

if __name__ == "__main__":
    minar_stats_pitchers()
