import os
import pandas as pd
import pybaseball

# 1. Activamos la caché para engañar un poco al servidor y hacer el proceso más estable
pybaseball.cache.enable()

TEMPORADAS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]

def extraer_sabermetria_mlb():
    print("⚾ [INICIO] Extrayendo métricas avanzadas de la MLB...")
    os.makedirs("data", exist_ok=True)
    
    df_bateo_total = pd.DataFrame()
    df_pitcheo_total = pd.DataFrame()

    print("📊 Intentando conectar con FanGraphs para datos de Bateo...")
    for year in TEMPORADAS:
        try:
            bateo = pybaseball.team_batting(year)
            if not bateo.empty:
                bateo['Season'] = year
                df_bateo_total = pd.concat([df_bateo_total, bateo], ignore_index=True)
                print(f"  -> {year} Bateo: {len(bateo)} equipos descargados.")
        except Exception as e:
            print(f"❌ Error crítico en Bateo {year}: {e}")

    print("📊 Intentando conectar con FanGraphs para datos de Pitcheo...")
    for year in TEMPORADAS:
        try:
            pitcheo = pybaseball.team_pitching(year)
            if not pitcheo.empty:
                pitcheo['Season'] = year
                df_pitcheo_total = pd.concat([df_pitcheo_total, pitcheo], ignore_index=True)
                print(f"  -> {year} Pitcheo: {len(pitcheo)} equipos descargados.")
        except Exception as e:
            print(f"❌ Error crítico en Pitcheo {year}: {e}")

    # Guardar los CSV maestros
    if not df_bateo_total.empty:
        ruta_bateo = "data/mlb_batting.csv"
        df_bateo_total.to_csv(ruta_bateo, index=False)
        print(f"✅ Bateo guardado exitosamente en {ruta_bateo}")
    else:
        print("⚠️ NO se generó el archivo de Bateo (bloqueo o datos vacíos).")
        
    if not df_pitcheo_total.empty:
        ruta_pitcheo = "data/mlb_pitching.csv"
        df_pitcheo_total.to_csv(ruta_pitcheo, index=False)
        print(f"✅ Pitcheo guardado exitosamente en {ruta_pitcheo}")
    else:
        print("⚠️ NO se generó el archivo de Pitcheo (bloqueo o datos vacíos).")

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
    extraer_sabermetria_mlb()
    generar_park_factors()
    print("🎯 Proceso finalizado.")
