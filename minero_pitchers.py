import os
import pandas as pd
from pybaseball import pitching_stats

def descargar_abridores_individuales():
    print("⚾ [INICIO] Extrayendo Sabermetría Individual de Pitchers (FanGraphs)...")
    os.makedirs("data", exist_ok=True)
    
    try:
        # Extraer lanzadores desde 2024 hasta 2026 (qual=20 filtra pitchers con menos de 20 innings)
        print("📊 Consultando servidores de FanGraphs. Esto puede tardar un par de minutos...")
        df_pitchers = pitching_stats(2024, 2026, qual=20)
        
        # Filtrar solo las columnas que consume tu app de Streamlit
        cols_necesarias = ['Name', 'Team', 'ERA', 'xFIP', 'GS']
        df_final = df_pitchers[cols_necesarias].copy()
        
        # GS = Games Started. Si GS > 0, significa que ha abierto juegos (es abridor)
        df_abridores = df_final[df_final['GS'] > 0].copy()
        
        # Opcional: Eliminar duplicados si un jugador lanzó para varios equipos en distintas temporadas, 
        # conservando su estadística más reciente.
        df_abridores = df_abridores.drop_duplicates(subset=['Name'], keep='first')
        
        # Guardar sobreescribiendo el archivo genérico
        ruta_archivo = "data/mlb_pitching.csv"
        df_abridores.to_csv(ruta_archivo, index=False)
        print(f"✅ [ÉXITO] Archivo '{ruta_archivo}' guardado con {len(df_abridores)} abridores reales.")
        
    except Exception as e:
        print(f"❌ Error crítico al descargar datos de FanGraphs: {e}")

if __name__ == "__main__":
    descargar_abridores_individuales()
