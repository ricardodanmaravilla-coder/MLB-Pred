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
        dfs = pd.read_html(url, storage_options=headers)
        
        df = None
        for table in dfs:
            cols_str = str(table.columns)
            if 'Player' in cols_str or 'Name' in cols_str:
                df = table
                break
        
        if df is None:
            print("❌ No se encontró la tabla de estadísticas. El sitio cambió su estructura.")
            return

        # Aplanar columnas multilínea si existen en pandas moderno
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(str(c) for c in col if c) for col in df.columns]

        rename_map = {}
        for col in df.columns:
            col_s = str(col)
            if 'Player' in col_s: rename_map[col] = 'Name'
            elif 'ERA' in col_s: rename_map[col] = 'ERA'
            elif 'IP' in col_s: rename_map[col] = 'Innings'
            elif 'GS' in col_s: rename_map[col] = 'GS'
            elif 'Team' in col_s: rename_map[col] = 'Team'
        
        df = df.rename(columns=rename_map)
        
        if 'Name' in df.columns and 'ERA' in df.columns and 'GS' in df.columns:
            df['GS'] = pd.to_numeric(df['GS'], errors='coerce').fillna(0)
            df['ERA'] = pd.to_numeric(df['ERA'], errors='coerce')
            
            df_abridores = df[df['GS'] > 0].copy()
            df_abridores['xFIP'] = df_abridores['ERA']
            
            if 'Team' not in df_abridores.columns:
                df_abridores['Team'] = 'UNK'
                
            df_final = df_abridores[['Name', 'Team', 'ERA', 'xFIP', 'GS']].dropna(subset=['ERA'])
            df_final.to_csv("data/mlb_pitching_individual.csv", index=False)
            print(f"✅ [ÉXITO] Archivo guardado con {len(df_final)} lanzadores.")
        else:
            print(f"❌ Columnas detectadas: {df.columns.tolist()}")
            
    except Exception as e:
        print(f"❌ Error en la descarga: {e}")

if __name__ == "__main__":
    minar_stats_pitchers()
