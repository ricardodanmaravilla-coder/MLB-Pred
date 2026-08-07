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
        if os.path.exists("data/mlb_batting.csv"):
            bateo = pd.read_csv("data/mlb_batting.csv", on_bad_lines='skip')
            bateo.columns = bateo.columns.str.strip()
            # Asegurar que Team y wRC+ existan y sean limpios
            if 'Team' in bateo.columns and 'wRC+' in bateo.columns:
                bateo['wRC+'] = pd.to_numeric(bateo['wRC+'], errors='coerce')

        if os.path.exists("data/mlb_pitching.csv"):
            pitcheo = pd.read_csv("data/mlb_pitching.csv", on_bad_lines='skip')
            pitcheo.columns = pitcheo.columns.str.strip()
            # Asegurar conversiones numéricas limpias para xFIP y ERA
            for col in ['xFIP', 'ERA']:
                if col in pitcheo.columns:
                    pitcheo[col] = pd.to_numeric(pitcheo[col], errors='coerce')

        if os.path.exists("data/mlb_park_factors.csv"): 
            park = pd.read_csv("data/mlb_park_factors.csv", on_bad_lines='skip')
            park.columns = park.columns.str.strip()
            
        if os.path.exists("data/mlb_games.csv"): 
            games = pd.read_csv("data/mlb_games.csv", on_bad_lines='skip')
            games.columns = games.columns.str.strip()
            
    except Exception as e:
        st.warning(f"Aviso de carga en archivos históricos: {e}")
        
    return bateo, pitcheo, park, games

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
            st.markdown("Escanea toda la cartelera buscando oportunidades con **probabilidad > 60%**.")
            
            if st.button("🚀 Ejecutar Escáner Global de la Jornada", type="primary"):
                with st.spinner("Escaneando duelos..."):
                    ml = PredictorMLMLB()
                    ml.entrenar(df_bat, df_pit, df_games)
                    
                    recomendaciones = []
                    
                    for llave, datos_partido in partidos_hoy.items():
                        loc_abbr = EQUIPOS_MAP.get(datos_partido["local"], "")
                        vis_abbr = EQUIPOS_MAP.get(datos_partido["visita"], "")
                        if not loc_abbr or not vis_abbr: continue
                        
                        try:
                            # --- CÁLCULO DE MÉTRICAS (Igual a tu lógica probada) ---
                            wrc_loc = float(df_bat[df_bat['Team'] == loc_abbr]['wRC+'].mean())
                            wrc_vis = float(df_bat[df_bat['Team'] == vis_abbr]['wRC+'].mean())
                            
                            # (Tu lógica de xFIP y Park Factors se mantiene igual...)
                            # ... [omitido aquí para brevedad, mantén tu código previo] ...
                            
                            # --- PREDICCIONES ---
                            preds_ml = ml.predecir_partido(loc_abbr, vis_abbr, wrc_loc, wrc_vis, xfip_loc, xfip_vis, park_factor)
                            res_mc = simular_partido_mlb(...) # (Tu función previa)

                            # --- EXTRACCIÓN SEGURA DE PROBABILIDADES ML ---
                            # Si 'Probabilidad_Local' no existe, intentamos buscar 'prob_local' o usar promedio
                            prob_ml_loc = preds_ml.get('Probabilidad_Local', preds_ml.get('prob_local', 50.0))
                            prob_ml_vis = preds_ml.get('Probabilidad_Visita', preds_ml.get('prob_visita', 50.0))
                            
                            prob_mc_loc = res_mc['Moneyline']['Gana Local']
                            prob_mc_vis = res_mc['Moneyline']['Gana Visita']
                            
                            # --- LÓGICA DE FILTRADO > 60% ---
                            # Ganador Local
                            if prob_mc_loc >= 60.0:
                                cuota = datos_partido["cuota_loc"]
                                kelly_pct = calcular_criterio_kelly(prob_mc_loc, cuota)
                                recomendaciones.append({
                                    "Partido": f"{datos_partido['visita']} @ {datos_partido['local']}",
                                    "Mercado": "Moneyline",
                                    "Apuesta": f"Gana Local ({datos_partido['local']})",
                                    "Prob. Montecarlo": f"{prob_mc_loc}%",
                                    "Prob. ML": f"{prob_ml_loc}%",
                                    "Stake Kelly": f"{kelly_pct}%"
                                })
                            # Ganador Visita
                            elif prob_mc_vis >= 60.0:
                                cuota = datos_partido["cuota_vis"]
                                kelly_pct = calcular_criterio_kelly(prob_mc_vis, cuota)
                                recomendaciones.append({
                                    "Partido": f"{datos_partido['visita']} @ {datos_partido['local']}",
                                    "Mercado": "Moneyline",
                                    "Apuesta": f"Gana Visita ({datos_partido['visita']})",
                                    "Prob. Montecarlo": f"{prob_mc_vis}%",
                                    "Prob. ML": f"{prob_ml_vis}%",
                                    "Stake Kelly": f"{kelly_pct}%"
                                })
                            
                            # Totales
                            carreras_dict = res_mc.get('Carreras', {})
                            prob_over = carreras_dict.get(f"Over {linea_casino}", 50.0)
                            if prob_over >= 60.0:
                                recomendaciones.append({
                                    "Partido": f"{datos_partido['visita']} @ {datos_partido['local']}",
                                    "Mercado": "Totales",
                                    "Apuesta": f"Over {linea_casino}",
                                    "Prob. Montecarlo": f"{prob_over}%",
                                    "Prob. ML": "N/A",
                                    "Stake Kelly": f"{calcular_criterio_kelly(prob_over, datos_partido.get('cuota_over', 1.91))}%"
                                })

                        except Exception as e:
                            continue
                    
                    if recomendaciones:
                        df_recom = pd.DataFrame(recomendaciones)
                        st.dataframe(df_recom, use_container_width=True)
                    else:
                        st.info("No se encontraron partidos con >60% de probabilidad hoy.")
