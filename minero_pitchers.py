import pandas as pd
import os
import requests
from io import StringIO

def minar_stats_pitchers():
    print("⚾ [INICIO] Extrayendo estadísticas REALES de lanzadores...")
    os.makedirs("data", exist_ok=True)
    
    url = "https://www.mlb.com/stats/pitching?playerPool=ALL&season=2026"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    try:
        print("📥 Conectando con mlb.com (modo robusto)...")
        # Usamos el encabezado para evitar bloqueos y pedimos las tablas
        dfs = pd.read_html(url, storage_options=headers)
        
        # En lugar de asumir que es la primera tabla (dfs[0]), 
        # buscamos la que tenga la columna 'Player' o similar
        df = None
        for i, table in enumerate(dfs):
            if 'Player' in table.columns or 'Name' in str(table.columns):
                df = table
                break
        
        if df is None:
            print("❌ No se encontró la tabla de estadísticas. El sitio cambió su estructura.")
            return

        # Limpieza inteligente: buscamos las columnas independientemente de cómo se llamen
        # Esto evita el error de "None of [Index] are in the columns"
        rename_map = {}
        for col in df.columns:
            if 'Player' in str(col): rename_map[col] = 'Name'
            if 'ERA' in str(col): rename_map[col] = 'ERA'
            if 'IP' in str(col): rename_map[col] = 'Innings'
            if 'GS' in str(col): rename_map[col] = 'GS'
        
        df = df.rename(columns=rename_map)
        
        # Filtrar columnas requeridas
        if all(c in df.columns for c in ['Name', 'ERA', 'GS']):
            df_abridores = df[df['GS'] > 0].copy()
            df_abridores['xFIP'] = df_abridores['ERA'] # Valor real como base
            
            df_abridores.to_csv("data/mlb_pitching_individual.csv", index=False)
            print(f"✅ [ÉXITO] Archivo guardado con {len(df_abridores)} lanzadores.")
        else:
            print(f"❌ Columnas detectadas: {df.columns.tolist()}")
            
    except Exception as e:
        print(f"❌ Error en la descarga: {e}") descarga: {e}")

if __name__ == "__main__":
    minar_stats_pitchers()
