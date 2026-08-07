import streamlit as st
import pandas as pd
import requests
import numpy as np
import os
ODDS_API_KEY = "de66554a17bce1149445b1a883056607"

from modules.montecarlo_mlb import simular_partido_mlb
from modules.ml_mlb import PredictorMLMLB
from modules.odds_mlb import analizar_apuestas_mlb

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="MLB Quant Analytics", layout="wide", page_icon="⚾")

# Mapeo de nombres de The Odds API / ESPN a las abreviaturas
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

# --- COORDENADAS DE LOS ESTADIOS DE LA MLB (Para el clima automático) ---
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
    """Consulta el clima actual en vivo mediante Open-Meteo (Gratis y sin Key)"""
    coords = ESTADIOS_COORDS.get(nombre_equipo, {"lat": 40.7128, "lon": -74.0060})
    url = f"https://api.open-meteo.com/v1/forecast?latitude={coords['lat']}&longitude={coords['lon']}&current=temperature_2m,wind_speed_10m,wind_direction_2m"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json().get('current', {})
            _temp_c = data.get('temperature_2m', 22.0)
            temp_f = int((_temp_c * 9/5) + 32)
            
            wind_kmh = data.get('wind_speed_10m', 8.0)
            wind_mph = int(wind_kmh * 0.621371)
            
            deg = data.get('wind_direction_2m', 0)
            dir_str = "None"
            if 45 <= deg < 135: dir_str = "Infield (Hacia Adentro)"
            elif 225 <= deg < 315: dir_str = "Outfield (Hacia Afuera)"
            
            return temp_f, wind_mph, dir_str
    except:
        pass
    return 72, 8, "None"

def american_to_decimal(am_odds):
    """Convierte cuotas americanas de Las Vegas a formato decimal europeo para la matemática"""
    if am_odds == 0: return 0.0
    if am_odds > 0: return round((am_odds / 100.0) + 1, 2)
    else: return round((100.0 / abs(am_odds)) + 1, 2)

# --- CARGA DE DATOS HISTÓRICOS ---
@st.cache_data(ttl=3600)
def cargar_datos_historicos():
    bateo = pd.DataFrame()
    pitcheo = pd.DataFrame()
    park = pd.DataFrame()
    games = pd.DataFrame()
    
    try:
        if os.path.exists("data/mlb_batting.csv"): 
            bateo = pd.read_csv("data/mlb_batting.csv")
        if os.path.exists("data/mlb_pitching.csv"): 
            pitcheo = pd.read_csv("data/mlb_pitching.csv")
        if os.path.exists("data/mlb_park_factors.csv"): 
            park = pd.read_csv("data/mlb_park_factors.csv")
        if os.path.exists("data/mlb_games.csv"): 
            games = pd.read_csv("data/mlb_games.csv")
    except Exception as e:
        st.warning(f"Aviso de carga de datos: {e}")
        
    return bateo, pitcheo, park, games

# Cargar los datos al inicio correctamente
df_bat, df_pit, df_parks, df_games = cargar_datos_historicos()
    
# Verificación rápida en la UI
if st.checkbox("Mostrar vista previa de los datos históricos"):
    st.write("Primeras 5 filas del historial de juegos:")
    st.dataframe(df_games.head())
    
@st.cache_data(ttl=300)
def obtener_cartelera_profesional():
    url = f"https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/?apiKey={ODDS_API_KEY}&regions=us&markets=h2h,totals&oddsFormat=american"
    partidos = {}
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            for game in data:
                home = game['home_team']
                away = game['away_team']
                
                for bookmaker in game['bookmakers']:
                    if bookmaker['key'] in ['draftkings', 'caesars', 'fanduel']:
                        markets = {m['key']: m['outcomes'] for m in bookmaker['markets']}
                        
                        h2h = markets.get('h2h', [])
                        linea_home = next((o['price'] for o in h2h if o['name'] == home), -110)
                        
                        totals = markets.get('totals', [])
                        linea_total = totals[0]['point'] if totals else 8.5
                        cuota_over = next((o['price'] for o in totals if o['name'] == 'Over'), -110)
                        
                        llave = f"⚾ {away} @ {home}"
                        partidos[llave] = {
                            "local": home, "visita": away,
                            "linea_carreras": linea_total,
                            "cuota_loc": american_to_decimal(linea_home),
                            "cuota_over": american_to_decimal(cuota_over)
                        }
                        break
        else:
            st.error(f"Error en Odds API: {res.status_code}")
    except Exception as e:
        st.error(f"Error conectando a Odds API: {e}")
    return partidos

# --- INTERFAZ PRINCIPAL ---
st.title("⚾ MLB Quant Analytics")
st.markdown("Motor predictivo basado en Sabermetría avanzada, simulaciones de Montecarlo y Machine Learning.")

if df_bat.empty or df_pit.empty:
    st.warning("⚠️ No se encontraron los datos históricos. Ejecuta primero `minero_mlb.py` para descargar la sabermetría.")
else:
    partidos_hoy = obtener_cartelera_profesional()
    
    if not partidos_hoy:
        st.info("No hay partidos programados o la API de Odds no retornó datos activos.")
    else:
        # 1. Selección del Partido
        st.subheader("1. Cartelera del Día (Vía The Odds API)")
        seleccion = st.selectbox("Selecciona un partido para analizar:", list(partidos_hoy.keys()))
        datos_partido = partidos_hoy[seleccion]
        
        # Obtener clima real automático basado en el estadio local
        temp_auto, viento_auto, dir_auto = obtener_clima_estadio(datos_partido["local"])
        
        # 2. Ajustes del Casino y Clima
        st.subheader("2. Condiciones del Casino y Clima (Auto-detectado)")
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            linea_carreras = st.number_input("Línea de Carreras (O/U)", value=float(datos_partido["linea_carreras"]), step=0.5)
            cuota_over = st.number_input("Cuota Casino (Over)", value=float(datos_partido["cuota_over"]), step=0.05)
        with c2:
            cuota_ml_local = st.number_input(f"Cuota ML ({datos_partido['local']})", value=float(datos_partido["cuota_loc"]), step=0.05)
        with c3:
            viento = st.number_input("Viento (mph)", value=viento_auto, step=1)
            dir_viento = st.selectbox("Dirección del Viento", ["None", "Outfield (Hacia Afuera)", "Infield (Hacia Adentro)"], index=["None", "Outfield (Hacia Afuera)", "Infield (Hacia Adentro)"].index(dir_auto) if dir_auto in ["None", "Outfield (Hacia Afuera)", "Infield (Hacia Adentro)"] else 0)
        with c4:
            temp = st.slider("Temperatura (°F)", min_value=30, max_value=110, value=temp_auto)
            
        # 3. Ejecución del Motor
        if st.button("🚀 Ejecutar Simulación Sniper", type="primary"):
            with st.spinner("Procesando Sabermetría y ejecutando 10,000 universos paralelos..."):
                
                # Mapear nombres
                loc_abbr = EQUIPOS_MAP.get(datos_partido["local"], "")
                vis_abbr = EQUIPOS_MAP.get(datos_partido["visita"], "")
                
                # Extraer estadísticas base (con fallbacks si falta algún dato)
                try: wrc_loc = float(df_bat[df_bat['Team'] == loc_abbr]['wRC+'].mean())
                except: wrc_loc = 100.0
                try: wrc_vis = float(df_bat[df_bat['Team'] == vis_abbr]['wRC+'].mean())
                except: wrc_vis = 100.0
                
                try: xfip_loc = float(df_pit[df_pit['Team'] == loc_abbr]['xFIP'].mean())
                except: xfip_loc = 4.10
                try: xfip_vis = float(df_pit[df_pit['Team'] == vis_abbr]['xFIP'].mean())
                except: xfip_vis = 4.10
                
                # Park Factor
                park_factor = 100.0
                altitud = 0.0
                if not df_parks.empty:
                    try:
                        park_data = df_parks[df_parks['Team'] == loc_abbr]
                        if not park_data.empty:
                            park_factor = float(park_data['Park_Factor_General'].values[0])
                            altitud = float(park_data['Altitud_pies'].values[0])
                    except: pass
                
                # MOTOR MACHINE LEARNING
                ml = PredictorMLMLB()
                ml.entrenar(df_bat, df_pit, df_games)
                preds_ml = ml.predecir_partido(loc_abbr, vis_abbr, wrc_loc, wrc_vis, xfip_loc, xfip_vis, park_factor)
                
                # MOTOR MONTECARLO (Pasándole la línea real del casino ajustada por el usuario)
                res_mc = simular_partido_mlb(
                    local=datos_partido['local'], visita=datos_partido['visita'],
                    pitcher_loc_xfip=xfip_loc, pitcher_vis_xfip=xfip_vis,
                    wrc_loc=wrc_loc, wrc_vis=wrc_vis,
                    bullpen_loc_era=xfip_loc * 1.05, bullpen_vis_era=xfip_vis * 1.05,
                    park_factor=park_factor, altitud_ft=altitud,
                    viento_mph=viento, direccion_viento=dir_viento, temp_f=temp,
                    linea_carreras_casino=linea_carreras,
                    num_simulaciones=10000
                )
                
                # EVALUACIÓN DE ODDS
                cuotas_reales = {
                    "Moneyline_Local": cuota_ml_local,
                    "Cuota_Over": cuota_over
                }
                df_apuestas = analizar_apuestas_mlb(res_mc, preds_ml, cuotas_reales, linea_carreras)
                
                # --- VISUALIZACIÓN DE RESULTADOS ---
                st.markdown("---")
                st.subheader(f"🏟️ Impacto del Estadio y Clima ({datos_partido['local']})")
                m1, m2, m3 = st.columns(3)
                m1.metric("Altitud", f"{altitud} ft")
                m2.metric("Park Factor", park_factor, delta="Favorece Bateo" if park_factor > 100 else "Favorece Pitcheo")
                m3.metric("Clima", f"{temp}°F | Viento: {viento}mph {dir_viento.split()[0]}")
                
                st.markdown("### 🎲 Probabilidades (Montecarlo 10k Simulaciones)")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric(f"Gana {datos_partido['local']}", f"{res_mc['Moneyline']['Gana Local']}%")
                c2.metric(f"Gana {datos_partido['visita']}", f"{res_mc['Moneyline']['Gana Visita']}%")
                c3.metric(f"Over {linea_carreras} Carreras", f"{res_mc['Carreras'][f'Over {linea_carreras}']}%")
                c4.metric(f"Promedio Hits Total", f"{res_mc['Hits']['Promedio_Total']}")
                
                st.markdown("### 🎯 Veredicto Financiero y Valor Esperado (EV+)")
                
                def color_veredicto(val):
                    if '🔥' in str(val): return 'color: #00ff00; font-weight: bold'
                    elif '✅' in str(val): return 'color: #adff2f'
                    elif '❌' in str(val): return 'color: #ff4d4d'
                    return ''
                    
                st.dataframe(
                    df_apuestas.style.map(color_veredicto, subset=['Veredicto']), 
                    use_container_width=True, hide_index=True
                )
