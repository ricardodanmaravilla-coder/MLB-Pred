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

def american_to_decimal(am_odds):
    """Convierte cuotas americanas de Las Vegas a formato decimal europeo automáticamente"""
    try:
        am_odds = float(am_odds)
        if am_odds == 0: return 1.91
        if am_odds > 0: return round((am_odds / 100.0) + 1, 2)
        else: return round((100.0 / abs(am_odds)) + 1, 2)
    except:
        return 1.91

def calcular_criterio_kelly(probabilidad_real, cuota_decimal, fraccion=0.25):
    """Calcula el porcentaje óptimo de bankroll a apostar usando el Criterio de Kelly Fraccionado"""
    try:
        p = float(probabilidad_real) / 100.0
        q = 1.0 - p
        b = float(cuota_decimal) - 1.0
        if b <= 0: return 0.0
        kelly = (b * p - q) / b
        # Se aplica Kelly fraccional (por defecto 1/4 o 0.25) para gestión de riesgo profesional en apuestas deportivas
        apuesta_recomendada = max(0.0, kelly * fraccion) * 100.0
        return round(apuesta_recomendada, 2)
    except:
        return 0.0

@st.cache_data(ttl=3600)
def cargar_datos_historicos():
    bateo, pitcheo, park, games = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    try:
        if os.path.exists("data/mlb_batting.csv"): bateo = pd.read_csv("data/mlb_batting.csv")
        if os.path.exists("data/mlb_pitching.csv"): pitcheo = pd.read_csv("data/mlb_pitching.csv")
        if os.path.exists("data/mlb_park_factors.csv"): 
            park = pd.read_csv("data/mlb_park_factors.csv")
            park.columns = park.columns.str.strip()
        if os.path.exists("data/mlb_games.csv"): games = pd.read_csv("data/mlb_games.csv")
    except Exception as e:
        st.warning(f"Aviso de carga: {e}")
    return bateo, pitcheo, park, games

df_bat, df_pit, df_parks, df_games = cargar_datos_historicos()

@st.cache_data(ttl=600)
def obtener_clima_estadio(nombre_equipo):
    ciudades = {
        "New York Yankees": "New_York", "Boston Red Sox": "Boston", "Los Angeles Dodgers": "Los_Angeles",
        "Houston Astros": "Houston", "Atlanta Braves": "Atlanta", "Philadelphia Phillies": "Philadelphia",
        "Baltimore Orioles": "Baltimore", "Tampa Bay Rays": "St_Petersburg", "Toronto Blue Jays": "Toronto",
        "Chicago White Sox": "Chicago", "Cleveland Guardians": "Cleveland", "Detroit Tigers": "Detroit",
        "Kansas City Royals": "Kansas_City", "Minnesota Twins": "Minneapolis", "Los Angeles Angels": "Anaheim",
        "Oakland Athletics": "Oakland", "Seattle Mariners": "Seattle", "Texas Rangers": "Arlington",
        "Chicago Cubs": "Chicago", "Cincinnati Reds": "Cincinnati", "Milwaukee Brewers": "Milwaukee",
        "Pittsburgh Pirates": "Pittsburgh", "St. Louis Cardinals": "St_Louis", "Arizona Diamondbacks": "Phoenix",
        "Colorado Rockies": "Denver", "San Francisco Giants": "San_Francisco", "San Diego Padres": "San_Diego",
        "Miami Marlins": "Miami", "New York Mets": "New_York", "Washington Nationals": "Washington"
    }
    
    ciudad = ciudades.get(nombre_equipo)
    if not ciudad:
        return None, None, "None"
        
    url = f"https://wttr.in/{ciudad}?format=j1"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            curr = data.get('current_condition', [{}])[0]
            temp_f = int(curr.get('temp_F'))
            wind_mph = int(curr.get('windspeedMiles'))
            wind_dir = curr.get('winddir16Point', '')
            
            dir_str = "None"
            if wind_dir in ['N', 'NNE', 'NNW', 'NE']: dir_str = "Infield (Hacia Adentro)"
            elif wind_dir in ['S', 'SSW', 'SSE', 'SW']: dir_str = "Outfield (Hacia Afuera)"
            elif wind_dir in ['E', 'ENE', 'ESE']: dir_str = "Lateral (Derecha a Izquierda)"
            elif wind_dir in ['W', 'WNW', 'WSW']: dir_str = "Lateral (Izquierda a Derecha)"
            
            return temp_f, wind_mph, dir_str
    except:
        pass
    return None, None, "None"

@st.cache_data(ttl=300)
def obtener_cartelera_y_cuotas_automaticas():
    hoy = datetime.date.today().strftime('%Y-%m-%d')
    url_mlb = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={hoy}&hydrate=probablePitcher,team"
    
    partidos = {}
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
                            "linea_carreras": 8.5,
                            "cuota_loc": 1.91, "cuota_vis": 1.91, "cuota_over": 1.91, "cuota_under": 1.91
                        }
    except Exception as e:
        st.error(f"Error en MLB StatsAPI: {e}")

    if ODDS_API_KEY != "TU_API_KEY_AQUI":
        url_odds = f"https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/?apiKey={ODDS_API_KEY}&regions=us&markets=h2h,totals&oddsFormat=american"
        try:
            res_odds = requests.get(url_odds, timeout=5)
            if res_odds.status_code == 200:
                data_odds = res_odds.json()
                for item in data_odds:
                    h_team = item.get('home_team')
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
st.markdown("Sistema autónomo de Sabermetría, Clima en Vivo, Duelo de Abridores, Escáner de Jornada y Criterio de Kelly.")

if df_bat.empty or df_pit.empty:
    st.warning("⚠️ Faltan datos históricos. Ejecuta `minero_mlb.py`.")
else:
    partidos_hoy = obtener_cartelera_y_cuotas_automaticas()
    if not partidos_hoy:
        st.info("No hay partidos programados para hoy.")
    else:
        modo_app = st.sidebar.radio("Modo de Operación", ["🎯 Análisis Individual por Partido", "🔍 Escáner Automático de la Jornada (EV+)"])
        
        if modo_app == "🔍 Escáner Automático de la Jornada (EV+)":
            st.subheader("🔍 Escáner Cuántico de Valor para Toda la Jornada")
            st.markdown("Escanea toda la cartelera buscando exclusivamente oportunidades con **probabilidades superiores al 60%** en simulaciones y calcula el monto exacto con el **Criterio de Kelly Fraccionado**.")
            
            if st.button("🚀 Ejecutar Escáner Global de la Jornada", type="primary"):
                with st.spinner("Escaneando duelos y filtrando alta probabilidad (>60%)..."):
                    ml = PredictorMLMLB()
                    ml.entrenar(df_bat, df_pit, df_games)
                    
                    recomendaciones = []
                    
                    for llave, datos_partido in partidos_hoy.items():
                        loc_abbr = EQUIPOS_MAP.get(datos_partido["local"], "")
                        vis_abbr = EQUIPOS_MAP.get(datos_partido["visita"], "")
                        if not loc_abbr or not vis_abbr: continue
                        
                        try:
                            wrc_loc = float(df_bat[df_bat['Team'] == loc_abbr]['wRC+'].mean())
                            wrc_vis = float(df_bat[df_bat['Team'] == vis_abbr]['wRC+'].mean())
                            
                            col_nombre_pitcher = 'Name'
                            for pc in ['Name', 'PlayerName', 'jugador', 'pitcher']:
                                if pc in df_pit.columns:
                                    col_nombre_pitcher = pc
                                    break
                            
                            p_loc_nom = datos_partido["pitcher_local"]
                            xfip_loc = None
                            if p_loc_nom != "Por Anunciar" and col_nombre_pitcher in df_pit.columns:
                                ml_match = df_pit[df_pit[col_nombre_pitcher].str.contains(p_loc_nom.split()[-1], case=False, na=False)]
                                if not ml_match.empty: xfip_loc = float(ml_match['xFIP'].values[0])
                            if xfip_loc is None: xfip_loc = float(df_pit[df_pit['Team'] == loc_abbr]['xFIP'].mean())

                            p_vis_nom = datos_partido["pitcher_visita"]
                            xfip_vis = None
                            if p_vis_nom != "Por Anunciar" and col_nombre_pitcher in df_pit.columns:
                                mv_match = df_pit[df_pit[col_nombre_pitcher].str.contains(p_vis_nom.split()[-1], case=False, na=False)]
                                if not mv_match.empty: xfip_vis = float(mv_match['xFIP'].values[0])
                            if xfip_vis is None: xfip_vis = float(df_pit[df_pit['Team'] == vis_abbr]['xFIP'].mean())

                            bullpen_loc_era = float(df_pit[df_pit['Team'] == loc_abbr]['ERA'].mean())
                            bullpen_vis_era = float(df_pit[df_pit['Team'] == vis_abbr]['ERA'].mean())
                            
                            park_data = df_parks[df_parks['Team'] == loc_abbr]
                            if park_data.empty: park_data = df_parks[df_parks.apply(lambda row: row.astype(str).str.contains(datos_partido["local"].split()[-1], case=False).any(), axis=1)]
                            if park_data.empty: continue
                            
                            col_pf = [c for c in park_data.columns if 'park_factor' in c.lower() or 'factor' in c.lower()][0]
                            col_alt = [c for c in park_data.columns if 'altitud' in c.lower() or 'alt' in c.lower()][0]
                            park_factor = float(park_data[col_pf].values[0])
                            altitud = float(park_data[col_alt].values[0])
                            
                            linea_casino = datos_partido["linea_carreras"] if datos_partido["linea_carreras"] is not None else 8.5
                            
                            # Ejecución exclusiva y robusta basada en Montecarlo
                            res_mc = simular_partido_mlb(
                                local=datos_partido['local'], visita=datos_partido['visita'],
                                pitcher_loc_xfip=xfip_loc, pitcher_vis_xfip=xfip_vis,
                                wrc_loc=wrc_loc, wrc_vis=wrc_vis,
                                bullpen_loc_era=bullpen_loc_era, bullpen_vis_era=bullpen_vis_era,
                                park_factor=park_factor, altitud_ft=altitud,
                                viento_mph=8, direccion_viento="None", temp_f=72,
                                linea_carreras_casino=linea_casino, num_simulaciones=200000
                            )
                            
                            # --- 1. EVALUACIÓN DE GANADOR (MONEYLINE) > 60% ---
                            prob_mc_loc = res_mc['Moneyline']['Gana Local']
                            prob_mc_vis = res_mc['Moneyline']['Gana Visita']
                            
                            if prob_mc_loc >= 60.0:
                                cuota = datos_partido["cuota_loc"]
                                kelly_pct = calcular_criterio_kelly(prob_mc_loc, cuota)
                                recomendaciones.append({
                                    "Partido": f"{datos_partido['visita']} @ {datos_partido['local']}",
                                    "Mercado": "Moneyline",
                                    "Apuesta Sugerida": f"Gana Local ({datos_partido['local']})",
                                    "Probabilidad Montecarlo": f"{prob_mc_loc}%",
                                    "Cuota Decimal": cuota,
                                    "Stake Kelly (%)": f"{kelly_pct}%"
                                })
                            elif prob_mc_vis >= 60.0:
                                cuota = datos_partido["cuota_vis"]
                                kelly_pct = calcular_criterio_kelly(prob_mc_vis, cuota)
                                recomendaciones.append({
                                    "Partido": f"{datos_partido['visita']} @ {datos_partido['local']}",
                                    "Mercado": "Moneyline",
                                    "Apuesta Sugerida": f"Gana Visita ({datos_partido['visita']})",
                                    "Probabilidad Montecarlo": f"{prob_mc_vis}%",
                                    "Cuota Decimal": cuota,
                                    "Stake Kelly (%)": f"{kelly_pct}%"
                                })
                                
                            # --- 2. EVALUACIÓN DE TOTALES (OVER / UNDER) > 60% ---
                            carreras_dict = res_mc.get('Carreras', {})
                            prob_over = carreras_dict.get(f"Over {linea_casino}", 50.0)
                            prob_under = carreras_dict.get(f"Under {linea_casino}", 50.0)
                            
                            if prob_over >= 60.0:
                                cuota_ov = datos_partido.get("cuota_over", 1.91)
                                kelly_ov = calcular_criterio_kelly(prob_over, cuota_ov)
                                recomendaciones.append({
                                    "Partido": f"{datos_partido['visita']} @ {datos_partido['local']}",
                                    "Mercado": f"Totales (O/U {linea_casino})",
                                    "Apuesta Sugerida": f"Over {linea_casino} Carreras",
                                    "Probabilidad Montecarlo": f"{prob_over}%",
                                    "Cuota Decimal": cuota_ov,
                                    "Stake Kelly (%)": f"{kelly_ov}%"
                                })
                            elif prob_under >= 60.0:
                                cuota_un = datos_partido.get("cuota_under", 1.91)
                                kelly_un = calcular_criterio_kelly(prob_under, cuota_un)
                                recomendaciones.append({
                                    "Partido": f"{datos_partido['visita']} @ {datos_partido['local']}",
                                    "Mercado": f"Totales (O/U {linea_casino})",
                                    "Apuesta Sugerida": f"Under {linea_casino} Carreras",
                                    "Probabilidad Montecarlo": f"{prob_under}%",
                                    "Cuota Decimal": cuota_un,
                                    "Stake Kelly (%)": f"{kelly_un}%"
                                })
                                
                        except Exception:
                            continue
                    
                    if recomendaciones:
                        st.success(f"🎯 ¡Se encontraron {len(recomendaciones)} oportunidades con más del 60% de probabilidad real!")
                        df_recom = pd.DataFrame(recomendaciones)
                        st.dataframe(df_recom, use_container_width=True, hide_index=True)
                    else:
                        st.info("ℹ️ Ningún encuentro de la cartelera actual supera el umbral estricto del 60% de probabilidad hoy. El mercado se encuentra sumamente disputado.")
