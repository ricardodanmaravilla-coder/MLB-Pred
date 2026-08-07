import os
import pandas as pd
from pybaseball import team_batting, team_pitching

# Usamos la temporada actual y la anterior para tener una muestra sólida
TEMPORADAS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]

def extraer_sabermetria_mlb():
    print("⚾ [INICIO] Extrayendo métricas avanzadas de la MLB...")
    os.makedirs("data", exist_ok=True)
    
    df_bateo_total = pd.DataFrame()
    df_pitcheo_total = pd.DataFrame()

    for year in TEMPORADAS:
        print(f"📊 Descargando datos de bateo y pitcheo del {year}...")
        try:
            # 1. Bateo: wRC+, OBP, SLG, BABIP, ISO (Poder aislado)
            bateo = team_batting(year)
            bateo['Season'] = year
            df_bateo_total = pd.concat([df_bateo_total, bateo], ignore_index=True)
            
            # 2. Pitcheo: FIP, xFIP, K/9, BB/9, LOB% (Control de daños)
            # Separamos abridores de relevistas usando el parámetro de pybaseball si está disponible, 
            # o extraemos el total del equipo como base.
            pitcheo = team_pitching(year)
            pitcheo['Season'] = year
            df_pitcheo_total = pd.concat([df_pitcheo_total, pitcheo], ignore_index=True)
            
        except Exception as e:
            print(f"⚠️ Error al descargar datos de {year}: {e}")

    # Guardar los CSV maestros
    if not df_bateo_total.empty:
        ruta_bateo = "data/mlb_batting.csv"
        df_bateo_total.to_csv(ruta_bateo, index=False)
        print(f"✅ Bateo guardado: {len(df_bateo_total)} registros en {ruta_bateo}")
        
    if not df_pitcheo_total.empty:
        ruta_pitcheo = "data/mlb_pitching.csv"
        df_pitcheo_total.to_csv(ruta_pitcheo, index=False)
        print(f"✅ Pitcheo guardado: {len(df_pitcheo_total)} registros en {ruta_pitcheo}")

def generar_park_factors():
    """
    Base de datos interna de Park Factors.
    100 es la media de la liga. > 100 favorece a bateadores (Over). < 100 favorece a pitchers (Under).
    Incluye el efecto de la altitud y dimensiones del estadio.
    """
    print("🏟️ Generando base de datos de Estadios (Park Factors & Altitud)...")
    
    # Muestra representativa de los factores (ajustados a métricas recientes)
    estadios = {
        "Team": ["COL", "CIN", "BOS", "LAA", "NYY", "ATL", "LAD", "HOU", "SD", "SF", "SEA"],
        "Estadio": ["Coors Field", "Great American", "Fenway Park", "Angel Stadium", "Yankee Stadium", 
                    "Truist Park", "Dodger Stadium", "Minute Maid", "Petco Park", "Oracle Park", "T-Mobile Park"],
        "Altitud_pies": [5200, 683, 15, 160, 54, 978, 267, 40, 13, 15, 10],
        "Park_Factor_General": [114, 107, 106, 103, 102, 101, 100, 99, 95, 94, 91],
        "Park_Factor_HR": [115, 128, 96, 114, 117, 105, 108, 97, 98, 86, 92] # Yankee y Cincinnati inflan HRs
    }
    
    df_park = pd.DataFrame(estadios)
    ruta_park = "data/mlb_park_factors.csv"
    df_park.to_csv(ruta_park, index=False)
    print(f"✅ Park Factors guardados en {ruta_park}")

if __name__ == "__main__":
    extraer_sabermetria_mlb()
    generar_park_factors()
    print("🎯 ¡Minería de datos MLB completada con éxito!")
