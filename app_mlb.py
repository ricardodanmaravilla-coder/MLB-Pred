import streamlit as st
import pandas as pd
import requests
import numpy as np
import os
import datetime

from modules.montecarlo_mlb import simular_partido_mlb
from modules.ml_mlb import PredictorMLMLB
from modules.odds_mlb import analizar_apuestas_mlb

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="MLB Quant Analytics", layout="wide", page_icon="⚾")

EQUIPOS_MAP = {
    "New York Yankees": "NYY", "Boston Red Sox": "BOS", "Los Angeles Dodgers": "LAD",
    "Houston Astros": "HOU", "Atlanta Braves": "ATL", "Philadelphia Phillies": "PHI",
    "Baltimore Orioles": "BAL", "Tampa Bay Rays": "TBR", "Toronto Blue Jays": "TOR",
    "Chicago White Sox": "CHW", "Cleveland Guardians": "CLE", "Detroit Tigers": "DET",
    "Kansas City Royals": "KCR", "Minnesota Twins": "MIN", "Los Angeles Angels": "LAA",
    "Oakland Athletics": "OAK", "Seattle Mariners": "SEA", "Texas Rangers": "TEX",
    "Chicago Cubs": "CHC", "Cincinnati Reds": "CIN", "Milwaukee Brewers": "MIL",
    "Pittsburgh Pirates": "PIT", "St. Louis Cardinals": "STL", "Arizona Diamondbacks": "ARI",
    "Colorado Rockies": "COL", "San Francisco Giants": "SFG", "San Diego Padres": "SDP",
    "Miami Marlins": "MIA", "New York Mets": "NYM", "Washington Nationals": "WSN"
}

ESTADIOS_COORDS = {
    "New York Yankees": {"lat": 40.8296, "lon": -73.9262},
    "Boston Red Sox": {"lat": 42.3467, "lon": -71.0972},
    "Los Angeles Dodgers": {"lat": 34.0739, "lon": -118.2400},
    "Houston Astros": {"lat": 29.7573, "lon": -95.3555},
    "Atlanta Braves": {"lat": 33.8907, "lon": -84.4678},
    "Philadelphia Phillies": {"lat": 39.9061, "lon": -75.1665},
    "Baltimore Orioles": {"lat": 39.2839, "lon": -76.6215},
    "Tampa Bay Rays": {"lat": 27.7682, "lon": -82.6534},
    "Toronto Blue Jays": {"lat": 43.6414, "lon": -79.3894},
    "Chicago White Sox": {"lat": 41.8299, "lon": -87.6338},
    "Cleveland Guardians": {"lat": 41.4962, "lon": -81.6852},
    "Detroit Tigers": {"lat": 42.3390, "lon": -83.0485},
    "Kansas City Royals": {"lat": 39.0517, "lon": -94.4803},
    "Minnesota Twins": {"lat": 44.9817, "lon": -93.2775},
    "Los Angeles Angels": {"lat": 33.8003, "lon": -117.8827},
    "Oakland Athletics": {"lat": 37.7516, "lon": -122.2005},
    "Seattle Mariners": {"lat": 47.5914, "lon": -123.3328},
    "Texas Rangers": {"lat": 32.7512, "lon": -97.0825},
    "Chicago Cubs": {"lat": 41.9484, "lon": -87.6553},
    "Cincinnati Reds": {"lat": 39.0973, "lon": -84.5068},
    "Milwaukee Brewers": {"lat": 43.0280, "lon": -87.9712},
    "Pittsburgh Pirates": {"lat": 40.4469, "lon": -80.0057},
    "St. Louis Cardinals": {"lat": 38.6226, "lon": -90.1928},
    "Arizona Diamondbacks": {"lat": 33.4455, "lon": -112.0667},
    "Colorado Rockies": {"lat": 39.7559, "lon": -104.9942},
    "San Francisco Giants": {"lat": 37.7786, "lon": -122.3893},
    "San Diego Padres": {"lat": 32.7076, "lon": -117.1570},
    "Miami Marlins": {"lat": 25.7781, "lon": -80.2196},
    "New York Mets": {"lat": 40.7571, "lon": -73.8458},
    "Washington Nationals": {"lat": 38.8730, "lon": -77.0074}
}

@st.cache_data(ttl=600)
def obtener_clima_estadio(nombre_equipo):
    coords = ESTADIOS_COORDS.get(nombre_equipo, {"lat": 40.7128, "lon": -74.0060})
    url = f"https://api.open-meteo.com/v1/forecast?latitude={coords['lat']}&longitude={coords['lon']}&current=temperature_2m,wind_speed_10m,wind_direction_2m"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json().get('current', {})
            temp_f = int((data.get('temperature_2m', 22.0) * 9/5) + 32)
            wind_mph = int(data.get('wind_speed_10m', 8.0) * 0.621371)
            deg = data.get('wind_direction_2m', 0)
            
            dir_str = "None"
            if (315 <= deg <= 360) or (0 <= deg < 45):
                dir_str = "Infield (Hacia Adentro)"
            elif 135 <= deg < 225:
                dir_str = "Outfield (Hacia Afuera)"
            return temp_f, wind_mph, dir_str
    except:
        pass
    return 72, 8, "None"

@st.cache_data(ttl=3600)
def cargar_datos_historicos():
    bateo, pitcheo, park, games = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    try:
        if os.path.exists("data/mlb_batting.csv"): bateo = pd.read_csv("data/mlb_batting.csv")
        if os.path.exists("data/mlb_pitching.csv"): pitcheo = pd.read_csv("data/mlb_pitching.csv")
        if os.path.exists("data/mlb_park_factors.csv"): park = pd.read_csv("data/mlb_park_factors.csv")
        if os.path.exists("data/mlb_games.csv"): games = pd.read_csv("data/mlb_games.csv")
    except Exception as e:
        st.warning(f"Aviso de carga: {e}")
    return bateo, pitcheo, park, games

df_bat, df_pit, df_parks, df_games = cargar_datos_historicos()

@st.cache_data(ttl=300)
def obtener_cartelera_mlb_oficial():
    hoy = datetime.date.today().strftime('%Y-%m-%d')
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={hoy}&hydrate=probablePitcher,team,odds"
    partidos = {}
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            for date_item in data.get('dates', []):
                for game in date_item.get('games', []):
                    home = game.get('teams', {}).get('home', {}).get('team', {}).get('name', '')
                    away = game.get('teams', {}).get('away', {}).get('team', {}).get('name', '')
                    
                    # Extraer abridores oficiales si la API los reporta
                    home_pitcher = game.get('teams', {}).get('home', {}).get('probablePitcher', {}).get('fullName', 'Por Anunciar')
                    away_pitcher = game.get('teams', {}).get('away', {}).get('probablePitcher', {}).get('fullName', 'Por Anunciar')
                    
                    linea_total = 8.5
                    odds_info = game.get('odds', [])
                    if odds_info:
                        try: linea_total = float(odds_info[0].get('overUnder', 8.5))
                        except: pass
                    
                    if home and away:
                        llave = f"⚾ {away} ({away_pitcher}) @ {home} ({home_pitcher})"
                        partidos[llave] = {
                            "local": home, "visita": away,
                            "pitcher_local": home_pitcher, "pitcher_visita": away_pitcher,
                            "linea_carreras": linea_total,
                            "cuota_loc": 1.91, "cuota_vis": 1.91, "cuota_over": 1.91
                        }
    except Exception as e:
        st.error(f"Error conectando a MLB StatsAPI: {e}")
    return partidos

# --- INTERFAZ ---
st.title("⚾ MLB Quant Analytics Pro")
st.markdown("Motor predictivo avanzado basado en Duelo de Abridores, Sabermetría y Clima Real.")

if df_bat.empty or df_pit.empty:
    st.warning("⚠️ Faltan datos históricos. Ejecuta `minero_mlb.py`.")
else:
    partidos_hoy = obtener_cartelera_mlb_oficial()
    if not partidos_hoy:
        st.info("No hay partidos programados para hoy en la API oficial de la MLB.")
    else:
        st.subheader("1. Cartelera Oficial del Día")
        seleccion = st.selectbox("Selecciona un duelo:", list(partidos_hoy.keys()))
        datos_partido = partidos_hoy[seleccion]
        
        temp_auto, viento_auto, dir_auto = obtener_clima_estadio(datos_partido["local"])
        
        st.subheader("2. Condiciones del Entorno y Líneas de Apuesta")
        c1, c2, c3, c4 = st.columns(4)
        
        opciones_viento = ["None", "Outfield (Hacia Afuera)", "Infield (Hacia Adentro)"]
        indice_dir = opciones_viento.index(dir_auto) if dir_auto in opciones_viento else 0
        
        with c1:
            linea_carreras = st.number_input("Línea O/U", value=float(datos_partido["linea_carreras"]), step=0.5)
            cuota_over = st.number_input("Cuota Over", value=1.91, step=0.01)
        with c2:
            cuota_ml_local = st.number_input(f"Cuota ML Local", value=1.91, step=0.01)
            cuota_ml_visita = st.number_input(f"Cuota ML Visita", value=1.91, step=0.01)
        with c3:
            viento = st.number_input("Viento (mph)", value=viento_auto, step=1)
            dir_viento = st.selectbox("Dirección del Viento", opciones_viento, index=indice_dir)
        with c4:
            temp = st.slider("Temperatura (°F)", 30, 110, temp_auto)
            
        if st.button("🚀 Ejecutar Simulación Cuántica", type="primary"):
            with st.spinner("Procesando duelo de lanzadores y simulando 500,000 escenarios..."):
                loc_abbr = EQUIPOS_MAP.get(datos_partido["local"], "")
                vis_abbr = EQUIPOS_MAP.get(datos_partido["visita"], "")
                
                # Obtención de Sabermetría real de los equipos
                wrc_loc = float(df_bat[df_bat['Team'] == loc_abbr]['wRC+'].mean()) if not df_bat.empty else 100.0
                wrc_vis = float(df_bat[df_bat['Team'] == vis_abbr]['wRC+'].mean()) if not df_bat.empty else 100.0
                
                # Búsqueda específica del pitcher o estimación basada en la rotación del equipo
                pitcher_loc_data = df_pit[df_pit['Team'] == loc_abbr]
                xfip_loc = float(pitcher_loc_data['xFIP'].mean()) if not pitcher_loc_data.empty else 4.10
                 bullpen_loc_era = xfip_loc * 1.05
                
                pitcher_vis_data = df_pit[df_pit['Team'] == vis_abbr]
                xfip_vis = float(pitcher_vis_data['xFIP'].mean()) if not pitcher_vis_data.empty else 4.10
                bullpen_vis_era = xfip_vis * 1.05
                
                # Park Factors reales
                park_factor, altitud = 100.0, 0.0
                if not df_parks.empty:
                    p_data = df_parks[df_parks['Team'] == loc_abbr]
                    if not p_data.empty:
                        park_factor = float(p_data['Park_Factor_General'].values[0])
                        altitud = float(p_data['Altitud_pies'].values[0])
                
                # Motor ML
                ml = PredictorMLMLB()
                ml.entrenar(df_bat, df_pit, df_games)
                preds_ml = ml.predecir_partido(loc_abbr, vis_abbr, wrc_loc, wrc_vis, xfip_loc, xfip_vis, park_factor)
                
                # Motor Montecarlo Real
                res_mc = simular_partido_mlb(
                    local=datos_partido['local'], visita=datos_partido['visita'],
                    pitcher_loc_xfip=xfip_loc, pitcher_vis_xfip=xfip_vis,
                    wrc_loc=wrc_loc, wrc_vis=wrc_vis,
                    bullpen_loc_era=bullpen_loc_era, bullpen_vis_era=bullpen_vis_era,
                    park_factor=park_factor, altitud_ft=altitud,
                    viento_mph=viento, direccion_viento=dir_viento, temp_f=temp,
                    linea_carreras_casino=linea_carreras,
                    num_simulaciones=500000
                )
                
                cuotas_reales = {
                    "Moneyline_Local": cuota_ml_local,
                    "Moneyline_Visita": cuota_ml_visita,
                    "Cuota_Over": cuota_over,
                    "Cuota_Under": cuota_over
                }
                df_apuestas = analizar_apuestas_mlb(res_mc, preds_ml, cuotas_reales, linea_carreras)
                
                # Resultados visuales
                st.markdown("---")
                st.subheader(f"🏟️ Factores Ambientales en {datos_partido['local']}")
                m1, m2, m3 = st.columns(3)
                m1.metric("Altitud del Parque", f"{altitud} ft")
                m2.metric("Park Factor General", park_factor)
                m3.metric("Clima en Vivo", f"{temp}°F | Viento: {viento}mph")
                
                st.markdown("### 🎲 Probabilidades Reales del Duelo")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric(f"Gana {datos_partido['local']}", f"{res_mc['Moneyline']['Gana Local']}%")
                c2.metric(f"Gana {datos_partido['visita']}", f"{res_mc['Moneyline']['Gana Visita']}%")
                
                prob_over = res_mc.get('Carreras', {}).get(f"Over {linea_carreras}", 50.0)
                c3.metric(f"Over {linea_carreras} Carreras", f"{prob_over}%")
                c4.metric("Promedio Carreras Total", f"{res_mc['Carreras']['Promedio_Total']}")
                
                st.markdown("### 🎯 Veredicto Financiero y Valor Esperado (EV+)")
                st.dataframe(df_apuestas, use_container_width=True, hide_index=True)
