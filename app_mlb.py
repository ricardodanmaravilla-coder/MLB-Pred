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

ODDS_API_KEY = "de66554a17bce1149445b1a883056607" 

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

def american_to_decimal(am_odds):
    """Convierte cuotas americanas de Las Vegas a formato decimal europeo automáticamente"""
    try:
        am_odds = float(am_odds)
        if am_odds == 0: return 1.91
        if am_odds > 0: return round((am_odds / 100.0) + 1, 2)
        else: return round((100.0 / abs(am_odds)) + 1, 2)
    except:
        return 1.91

@st.cache_data(ttl=600)
def obtener_clima_estadio(nombre_equipo):
    # 1. Normalizar nombres de columnas eliminando espacios accidentales
    df_parks.columns = df_parks.columns.str.strip()
    
    # 2. Buscar la abreviatura del equipo
    abbr = EQUIPOS_MAP.get(nombre_equipo, "")
    
    # 3. Buscar datos en el CSV
    park_data = df_parks[df_parks['Team'] == abbr]
    
    if park_data.empty:
        return None, None, "No disponible"
    
    # 4. Extracción dinámica: busca 'Latitud' o 'Lat' o 'Latitude'
    # Esto evita el KeyError si el nombre en el CSV es ligeramente diferente
    try:
        lat_col = [c for c in park_data.columns if 'lat' in c.lower()][0]
        lon_col = [c for c in park_data.columns if 'lon' in c.lower()][0]
        
        lat = float(park_data[lat_col].values[0])
        lon = float(park_data[lon_col].values[0])
    except (IndexError, KeyError, ValueError):
        return None, None, "Error en columnas de Lat/Lon"
    
    # 5. Consulta a la API
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,wind_speed_10m,wind_direction_2m"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json().get('current', {})
            temp_f = int((data.get('temperature_2m', 22.0) * 9/5) + 32)
            wind_mph = int(data.get('wind_speed_10m', 8.0) * 0.621371)
            deg = data.get('wind_direction_2m', 0)
            
            dir_str = "None"
            if (315 <= deg <= 360) or (0 <= deg < 45): dir_str = "Infield (Hacia Adentro)"
            elif 135 <= deg < 225: dir_str = "Outfield (Hacia Afuera)"
            return temp_f, wind_mph, dir_str
    except:
        pass
    return None, None, "Error API"
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
def obtener_cartelera_y_cuotas_automaticas():
    """Consulta la API de Odds y la API de MLB para extraer cuotas y abridores reales de forma dinámica"""
    hoy = datetime.date.today().strftime('%Y-%m-%d')
    url_mlb = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={hoy}&hydrate=probablePitcher,team"
    
    partidos = {}
    
    # 1. Obtener cartelera y abridores de la API oficial
    try:
        res_mlb = requests.get(url_mlb, timeout=5)
        if res_mlb.status_code == 200:
            data_mlb = res_mlb.json()
            for date_item in data_mlb.get('dates', []):
                for game in date_item.get('games', []):
                    home = game.get('teams', {}).get('home', {}).get('team', {}).get('name', '')
                    away = game.get('teams', {}).get('away', {}).get('team', {}).get('name', '')
                    home_pitcher = game.get('teams', {}).get('home', {}).get('probablePitcher', {}).get('fullName', 'Por Anunciar')
                    away_pitcher = game.get('teams', {}).get('away', {}).get('probablePitcher', {}).get('fullName', 'Por Anunciar')
                    
                    if home and away:
                        llave = f"⚾ {away} ({away_pitcher}) @ {home} ({home_pitcher})"
                        partidos[llave] = {
                            "local": home, "visita": away,
                            "pitcher_local": home_pitcher, "pitcher_visita": away_pitcher,
                            "linea_carreras": None,
                            "cuota_loc": None, "cuota_vis": None, "cuota_over": None, "cuota_under": None
                        }
    except Exception as e:
        st.error(f"Error en MLB StatsAPI: {e}")

    # 2. Obtener cuotas reales automáticas de The Odds API
    if ODDS_API_KEY != "de66554a17bce1149445b1a883056607":
        url_odds = f"https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/?apiKey={ODDS_API_KEY}&regions=us&markets=h2h,totals&oddsFormat=american"
        try:
            res_odds = requests.get(url_odds, timeout=5)
            if res_odds.status_code == 200:
                data_odds = res_odds.json()
                for item in data_odds:
                    h_team = item.get('home_team')
                    a_team = item.get('away_team')
                    for k, p in partidos.items():
                        if p['local'].lower() in h_team.lower() or h_team.lower() in p['local'].lower():
                            bookmakers = item.get('bookmakers', [])
                            if bookmakers:
                                markets = bookmakers[0].get('markets', [])
                                for m in markets:
                                    if m['key'] == 'h2h':
                                        for out in m['outcomes']:
                                            if out['name'] == h_team:
                                                p['cuota_loc'] = american_to_decimal(out['price'])
                                            else:
                                                p['cuota_vis'] = american_to_decimal(out['price'])
                                    elif m['key'] == 'totals':
                                        for out in m['outcomes']:
                                            if out['name'] == 'Over':
                                                p['linea_carreras'] = float(out['point'])
                                                p['cuota_over'] = american_to_decimal(out['price'])
                                            elif out['name'] == 'Under':
                                                p['cuota_under'] = american_to_decimal(out['price'])
        except Exception as e:
            st.warning(f"Aviso de sincronización de cuotas: {e}")
            
    return partidos

# --- INTERFAZ ---
st.title("⚾ MLB Quant Analytics Pro (Automático)")
st.markdown("Sistema autónomo de Sabermetría, Clima en Vivo, Duelo de Abridores y Cuotas de Mercado Reales.")

if df_bat.empty or df_pit.empty:
    st.warning("⚠️ Faltan datos históricos. Ejecuta `minero_mlb.py`.")
else:
    partidos_hoy = obtener_cartelera_y_cuotas_automaticas()
    if not partidos_hoy:
        st.info("No hay partidos programados para hoy.")
    else:
        st.subheader("1. Cartelera Oficial Sincronizada")
        seleccion = st.selectbox("Selecciona un duelo:", list(partidos_hoy.keys()))
        datos_partido = partidos_hoy[seleccion]
        
        # 1. Obtenemos los valores desde la API
        temp_auto, viento_auto, dir_auto = obtener_clima_estadio(datos_partido["local"])
        
        # 2. AQUI ES DONDE VA EL BLOQUE QUE ME MOSTRASTE
        st.subheader("2. Datos del Mercado y Clima (En Vivo)")
        c1, c2, c3, c4 = st.columns(4)
        
        opciones_viento = ["None", "Outfield (Hacia Afuera)", "Infield (Hacia Adentro)", "Lateral (Derecha a Izquierda)", "Lateral (Izquierda a Derecha)"]
        indice_dir = opciones_viento.index(dir_auto) if dir_auto in opciones_viento else 0
        
        with c1:
            st.metric("Línea O/U Casino", datos_partido["linea_carreras"] if datos_partido["linea_carreras"] is not None else "No disponible")
            st.metric("Cuota Over", datos_partido["cuota_over"] if datos_partido["cuota_over"] is not None else "No disponible")
        with c2:
            st.metric(f"Cuota ML ({datos_partido['local']})", datos_partido["cuota_loc"] if datos_partido["cuota_loc"] is not None else "No disponible")
            st.metric(f"Cuota ML ({datos_partido['visita']})", datos_partido["cuota_vis"] if datos_partido["cuota_vis"] is not None else "No disponible")
        with c3:
            viento = st.number_input("Viento (mph)", value=int(viento_auto) if viento_auto is not None else 8, step=1)
            dir_viento = st.selectbox("Dirección del Viento", opciones_viento, index=indice_dir)
        with c4:
            temp = st.slider("Temperatura (°F)", 30, 110, int(temp_auto) if temp_auto is not None else 72)
            
        if st.button("🚀 Ejecutar Simulación Cuántica", type="primary"):
            with st.spinner("Procesando datos en vivo y ejecutando 500,000 escenarios..."):
                loc_abbr = EQUIPOS_MAP.get(datos_partido["local"], "")
                vis_abbr = EQUIPOS_MAP.get(datos_partido["visita"], "")
                
                if df_bat.empty or df_pit.empty or df_parks.empty:
                    st.error("❌ Error crítico: Las bases de datos históricas están vacías. Ejecuta 'minero_mlb.py'.")
                    st.stop()

                try:
                    wrc_loc = float(df_bat[df_bat['Team'] == loc_abbr]['wRC+'].mean())
                    wrc_vis = float(df_bat[df_bat['Team'] == vis_abbr]['wRC+'].mean())
                except Exception as e:
                    st.error(f"Error procesando wRC+ de bateo: {e}")
                    st.stop()
                
                # Detección dinámica de columna de nombre de pitcher
                col_nombre_pitcher = 'Name'
                for posible_col in ['Name', 'PlayerName', 'jugador', 'pitcher']:
                    if posible_col in df_pit.columns:
                        col_nombre_pitcher = posible_col
                        break

                # Búsqueda Quirúrgica del Abridor Local
                pitcher_loc_nombre = datos_partido["pitcher_local"]
                xfip_loc = None
                
                if pitcher_loc_nombre != "Por Anunciar" and col_nombre_pitcher in df_pit.columns:
                    match_loc = df_pit[df_pit[col_nombre_pitcher].str.contains(pitcher_loc_nombre.split()[-1], case=False, na=False)]
                    if not match_loc.empty:
                        xfip_loc = float(match_loc['xFIP'].values[0])
                
                if xfip_loc is None:
                    team_pit_loc = df_pit[df_pit['Team'] == loc_abbr]
                    xfip_loc = float(team_pit_loc['xFIP'].mean())

                # Búsqueda Quirúrgica del Abridor Visitante
                pitcher_vis_nombre = datos_partido["pitcher_visita"]
                xfip_vis = None
                
                if pitcher_vis_nombre != "Por Anunciar" and col_nombre_pitcher in df_pit.columns:
                    match_vis = df_pit[df_pit[col_nombre_pitcher].str.contains(pitcher_vis_nombre.split()[-1], case=False, na=False)]
                    if not match_vis.empty:
                        xfip_vis = float(match_vis['xFIP'].values[0])
                
                if xfip_vis is None:
                    team_pit_vis = df_pit[df_pit['Team'] == vis_abbr]
                    xfip_vis = float(team_pit_vis['xFIP'].mean())

                bullpen_loc_era = float(df_pit[df_pit['Team'] == loc_abbr]['ERA'].mean())
                bullpen_vis_era = float(df_pit[df_pit['Team'] == vis_abbr]['ERA'].mean())
                
                # 4. Factores de Estadio (Búsqueda estricta en el CSV sin valores fijos inventados)
                if df_parks.empty:
                    st.error("❌ Error crítico: El archivo de factores de estadios (mlb_park_factors.csv) está vacío.")
                    st.stop()

                # Detectar automáticamente las columnas del CSV de parques
                col_team_park = 'Team'
                for posible_col in ['Team', 'team', 'Equipo', 'franchise']:
                    if posible_col in df_parks.columns:
                        col_team_park = posible_col
                        break

                park_data = df_parks[df_parks[col_team_park] == loc_abbr]
                
                if park_data.empty:
                    st.error(f"❌ No se encontró el registro para el equipo '{loc_abbr}' en el archivo de factores de estadios. Revisa las abreviaturas en tu CSV.")
                    st.stop()

                # Detectar columnas de métricas de parque de forma dinámica
                col_pf = 'Park_Factor_General' if 'Park_Factor_General' in park_data.columns else park_data.columns[1]
                col_alt = 'Altitud_pies' if 'Altitud_pies' in park_data.columns else park_data.columns[2]

                park_factor = float(park_data[col_pf].values[0])
                altitud = float(park_data[col_alt].values[0])
                
                linea_casino = datos_partido["linea_carreras"] if datos_partido["linea_carreras"] is not None else 8.5
                
                ml = PredictorMLMLB()
                ml.entrenar(df_bat, df_pit, df_games)
                preds_ml = ml.predecir_partido(loc_abbr, vis_abbr, wrc_loc, wrc_vis, xfip_loc, xfip_vis, park_factor)
                
                res_mc = simular_partido_mlb(
                    local=datos_partido['local'], visita=datos_partido['visita'],
                    pitcher_loc_xfip=xfip_loc, pitcher_vis_xfip=xfip_vis,
                    wrc_loc=wrc_loc, wrc_vis=wrc_vis,
                    bullpen_loc_era=bullpen_loc_era, bullpen_vis_era=bullpen_vis_era,
                    park_factor=park_factor, altitud_ft=altitud,
                    viento_mph=viento, direccion_viento=dir_viento, temp_f=temp,
                    linea_carreras_casino=linea_casino,
                    num_simulaciones=500000
                )
                
                cuotas_reales = {
                    "Moneyline_Local": datos_partido["cuota_loc"] if datos_partido["cuota_loc"] is not None else 1.91,
                    "Moneyline_Visita": datos_partido["cuota_vis"] if datos_partido["cuota_vis"] is not None else 1.91,
                    "Cuota_Over": datos_partido["cuota_over"] if datos_partido["cuota_over"] is not None else 1.91,
                    "Cuota_Under": datos_partido["cuota_under"] if datos_partido["cuota_under"] is not None else 1.91
                }
                df_apuestas = analizar_apuestas_mlb(res_mc, preds_ml, cuotas_reales, linea_casino)
                
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
                
                prob_over = res_mc.get('Carreras', {}).get(f"Over {linea_casino}", 50.0)
                c3.metric(f"Over {linea_casino} Carreras", f"{prob_over}%")
                c4.metric("Promedio Carreras Total", f"{res_mc['Carreras']['Promedio_Total']}")
                
                st.markdown("### 🎯 Veredicto Financiero y Valor Esperado (EV+)")
                st.dataframe(df_apuestas, use_container_width=True, hide_index=True)
