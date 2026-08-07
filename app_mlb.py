import streamlit as st
import pandas as pd
import requests
import numpy as np

from modules.montecarlo_mlb import simular_partido_mlb
from modules.ml_mlb import PredictorMLMLB
from modules.odds_mlb import analizar_apuestas_mlb

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="MLB Quant Analytics", layout="wide", page_icon="⚾")

# Mapeo de nombres de ESPN a las abreviaturas de pybaseball
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

def american_to_decimal(am_odds):
    """Convierte cuotas americanas de Las Vegas a formato decimal europeo para la matemática"""
    if am_odds == 0: return 0.0
    if am_odds > 0: return round((am_odds / 100.0) + 1, 2)
    else: return round((100.0 / abs(am_odds)) + 1, 2)

@st.cache_data
def cargar_datos_historicos():
    try:
        batting = pd.read_csv("data/mlb_batting.csv")
        pitching = pd.read_csv("data/mlb_pitching.csv")
        parks = pd.read_csv("data/mlb_park_factors.csv")
        return batting, pitching, parks
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=300)
def obtener_cartelera_espn():
    url = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
    partidos = {}
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            for event in data.get('events', []):
                comp = event.get('competitions', [{}])[0]
                estado = comp.get('status', {}).get('type', {}).get('state', '')
                
                # Solo tomamos partidos que aún no terminan
                if estado == 'post': continue
                
                local, visita = "", ""
                for team in comp.get('competitors', []):
                    nombre = team.get('team', {}).get('displayName', '')
                    if team.get('homeAway') == 'home': local = nombre
                    else: visita = nombre
                    
                # Extraer Cuotas (Odds) dinámicas de ESPN
                linea_carreras = 8.5
                cuota_loc_dec = 1.90
                cuota_over_dec = 1.90
                
                odds = comp.get('odds', [])
                if odds:
                    main_odds = odds[0]
                    linea_carreras = main_odds.get('overUnder', 8.5)
                    loc_odds = main_odds.get('homeTeamOdds', {}).get('moneyLine', -110)
                    over_odds = main_odds.get('overOdds', -110)
                    
                    cuota_loc_dec = american_to_decimal(loc_odds)
                    cuota_over_dec = american_to_decimal(over_odds)

                if local and visita:
                    llave = f"⚾ {visita} @ {local}"
                    partidos[llave] = {
                        "local": local, "visita": visita,
                        "linea_carreras": linea_carreras,
                        "cuota_loc": cuota_loc_dec, "cuota_over": cuota_over_dec
                    }
    except Exception as e:
        st.error(f"Error conectando a ESPN: {e}")
    return partidos

# --- INTERFAZ PRINCIPAL ---
st.title("⚾ MLB Quant Analytics")
st.markdown("Motor predictivo basado en Sabermetría avanzada, simulaciones de Montecarlo y Machine Learning.")

df_batting, df_pitching, df_parks = cargar_datos_historicos()

if df_batting.empty or df_pitching.empty:
    st.warning("⚠️ No se encontraron los datos históricos. Ejecuta primero `minero_mlb.py` para descargar la sabermetría.")
else:
    partidos_hoy = obtener_cartelera_espn()
    
    if not partidos_hoy:
        st.info("No hay partidos programados o la API de ESPN no retornó datos.")
    else:
        # 1. Selección del Partido
        st.subheader("1. Cartelera del Día (Vía ESPN)")
        seleccion = st.selectbox("Selecciona un partido para analizar:", list(partidos_hoy.keys()))
        datos_partido = partidos_hoy[seleccion]
        
        # 2. Ajustes del Casino y Clima
        st.subheader("2. Condiciones del Casino y Variables del Entorno")
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            linea_carreras = st.number_input("Línea de Carreras (O/U)", value=float(datos_partido["linea_carreras"]), step=0.5)
            cuota_over = st.number_input("Cuota Casino (Over)", value=float(datos_partido["cuota_over"]), step=0.05)
        with c2:
            cuota_ml_local = st.number_input(f"Cuota ML ({datos_partido['local']})", value=float(datos_partido["cuota_loc"]), step=0.05)
        with c3:
            viento = st.number_input("Viento (mph)", value=5, step=1)
            dir_viento = st.selectbox("Dirección del Viento", ["None", "Outfield (Hacia Afuera)", "Infield (Hacia Adentro)"])
        with c4:
            temp = st.slider("Temperatura (°F)", min_value=30, max_value=110, value=72)
            
        # 3. Ejecución del Motor
        if st.button("🚀 Ejecutar Simulación Sniper", type="primary"):
            with st.spinner("Procesando Sabermetría y ejecutando 10,000 universos paralelos..."):
                
                # Mapear nombres
                loc_abbr = EQUIPOS_MAP.get(datos_partido["local"], "")
                vis_abbr = EQUIPOS_MAP.get(datos_partido["visita"], "")
                
                # Extraer estadísticas base (con fallbacks si falta algún dato)
                try: wrc_loc = float(df_batting[df_batting['Team'] == loc_abbr]['wRC+'].mean())
                except: wrc_loc = 100.0
                try: wrc_vis = float(df_batting[df_batting['Team'] == vis_abbr]['wRC+'].mean())
                except: wrc_vis = 100.0
                
                try: xfip_loc = float(df_pitching[df_pitching['Team'] == loc_abbr]['xFIP'].mean())
                except: xfip_loc = 4.10
                try: xfip_vis = float(df_pitching[df_pitching['Team'] == vis_abbr]['xFIP'].mean())
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
                ml_engine = PredictorMLMLB()
                ml_engine.entrenar(df_batting, df_pitching)
                preds_ml = ml_engine.predecir_partido(wrc_loc, wrc_vis, xfip_loc, xfip_vis, park_factor)
                
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
