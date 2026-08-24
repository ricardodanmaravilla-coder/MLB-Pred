import streamlit as st
import pandas as pd
import requests
import numpy as np
import os
import datetime
import math

from modules.montecarlo_mlb import simular_partido_mlb
from modules.odds_mlb import analizar_apuestas_mlb
from modules.ml_mlb import PredictorMLMLB

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="MLB Quant Analytics", layout="wide", page_icon="⚾")

# Prefer a private runtime secret. Keep the legacy value only as a temporary
# compatibility fallback so the stable app does not lose live odds during migration.
def _get_odds_api_key():
    key = os.getenv("ODDS_API_KEY", "").strip()
    if key:
        return key
    try:
        key = str(st.secrets.get("ODDS_API_KEY", "")).strip()
        if key:
            return key
    except Exception:
        pass
    return "f9ffe1d7530a88b08e853659466c46ff"

ODDS_API_KEY = _get_odds_api_key()

EQUIPOS_MAP = {
    "New York Yankees": "NYY", "Boston Red Sox": "BOS", "Los Angeles Dodgers": "LAD",
    "Houston Astros": "HOU", "Atlanta Braves": "ATL", "Philadelphia Phillies": "PHI",
    "Baltimore Orioles": "BAL", "Tampa Bay Rays": "TB", "Toronto Blue Jays": "TOR",
    "Chicago White Sox": "CWS", "Cleveland Guardians": "CLE", "Detroit Tigers": "DET",
    "Kansas City Royals": "KC", "Minnesota Twins": "MIN", "Los Angeles Angels": "LAA",
    "Oakland Athletics": "OAK", "Athletics": "OAK", "Sacramento Athletics": "OAK", # Actualizado por reubicación
    "Seattle Mariners": "SEA", "Texas Rangers": "TEX", "Chicago Cubs": "CHC", 
    "Cincinnati Reds": "CIN", "Milwaukee Brewers": "MIL", "Pittsburgh Pirates": "PIT", 
    "St. Louis Cardinals": "STL", "Arizona Diamondbacks": "AZ", "Colorado Rockies": "COL", 
    "San Francisco Giants": "SF", "San Diego Padres": "SD", "Miami Marlins": "MIA", 
    "New York Mets": "NYM", "Washington Nationals": "WSH"
}

def american_to_decimal(am_odds):
    """Convierte cuotas americanas de Las Vegas a formato decimal europeo automáticamente"""
    try:
        if am_odds is None: return None
        am_odds = float(am_odds)
        if am_odds == 0: return None
        if am_odds > 0: return round((am_odds / 100.0) + 1, 2)
        else: return round((100.0 / abs(am_odds)) + 1, 2)
    except:
        return None

def _prob_no_vig_dos_vias(cuota_a, cuota_b):
    try:
        a, b = float(cuota_a), float(cuota_b)
        if a <= 1 or b <= 1:
            return None, None
        ia, ib = 1.0 / a, 1.0 / b
        total = ia + ib
        return ia / total, ib / total
    except (TypeError, ValueError, ZeroDivisionError):
        return None, None

def _pasa_valor(prob_pct, cuota, mercado_no_vig=None, min_ev=0.03, min_edge=0.025):
    try:
        p = float(prob_pct) / 100.0
        o = float(cuota)
        ev = p * o - 1.0
        if ev < min_ev:
            return False
        if mercado_no_vig is not None and (p - float(mercado_no_vig)) < min_edge:
            return False
        return True
    except (TypeError, ValueError):
        return False

def _score_valor(prob_pct, cuota, mercado_no_vig=None, desacuerdo_pp=0.0):
    """Rank candidates by market-relative value, not raw probability.

    This prevents naturally high-base-rate markets such as +1.5 from
    dominating merely because their nominal win probability is larger.
    """
    try:
        p = float(prob_pct) / 100.0
        o = float(cuota)
        ev_pct = (p * o - 1.0) * 100.0
        edge_pp = 0.0 if mercado_no_vig is None else (p - float(mercado_no_vig)) * 100.0
        disagreement_penalty = max(0.0, float(desacuerdo_pp)) * 0.15
        return round((1.5 * edge_pp) + ev_pct - disagreement_penalty, 4)
    except (TypeError, ValueError):
        return -999.0


def calcular_criterio_kelly(probabilidad_real, cuota_decimal, fraccion=0.25):
    """Calcula el porcentaje óptimo de bankroll a apostar usando el Criterio de Kelly Fraccionado"""
    try:
        if cuota_decimal is None or probabilidad_real is None: return 0.0
        p = float(probabilidad_real) / 100.0
        q = 1.0 - p
        b = float(cuota_decimal) - 1.0
        if b <= 0: return 0.0
        kelly = (b * p - q) / b
        apuesta_recomendada = max(0.0, kelly * fraccion) * 100.0
        return round(apuesta_recomendada, 2)
    except:
        return 0.0

def estimar_prob_ml(proyeccion, linea, tipo="over", sigma=None):
    """
    Convierte la proyección del Machine Learning en probabilidad real (%)
    utilizando la Función de Distribución Acumulada (CDF) Normal.
    """
    if proyeccion is None or linea is None:
        return 50.0

    # Sigma comes from chronological out-of-sample residuals when available.
    try:
        sigma = float(sigma) if sigma is not None else (3.5 if tipo in ["over", "under"] else 4.2)
        sigma = max(1.0, sigma)
        if tipo == "over":
            z = (proyeccion - linea) / sigma
        elif tipo == "under":
            z = (linea - proyeccion) / sigma
        elif tipo in ["spread_loc", "spread_vis"]:
            z = (proyeccion + linea) / sigma
        else:
            return 50.0

        prob = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        return max(0.0, min(100.0, round(prob * 100.0, 2)))
        
    except Exception as e:
        print(f"Error en CDF: {e}")
        return 50.0


@st.cache_data(ttl=3600)
def cargar_datos_historicos():
    bateo = pd.DataFrame()
    pitcheo = pd.DataFrame()
    pitcheo_individual = pd.DataFrame()
    park = pd.DataFrame()
    games = pd.DataFrame()

    try:
        if os.path.exists("data/mlb_batting.csv"):
            df_temp = pd.read_csv("data/mlb_batting.csv", sep=None, engine='python', on_bad_lines='skip')
            df_temp.columns = df_temp.columns.str.strip()
            if not df_temp.empty and 'Team' in df_temp.columns and 'wRC+' in df_temp.columns:
                df_temp['wRC+'] = pd.to_numeric(df_temp['wRC+'], errors='coerce')
                bateo = df_temp.dropna(subset=['wRC+'])

        if os.path.exists("data/mlb_pitching.csv"):
            df_temp = pd.read_csv("data/mlb_pitching.csv", sep=None, engine='python', on_bad_lines='skip')
            df_temp.columns = df_temp.columns.str.strip()
            if not df_temp.empty and 'Team' in df_temp.columns and 'xFIP' in df_temp.columns:
                df_temp['xFIP'] = pd.to_numeric(df_temp['xFIP'], errors='coerce')
                df_temp['ERA'] = pd.to_numeric(df_temp['ERA'], errors='coerce')
                pitcheo = df_temp.dropna(subset=['xFIP', 'ERA'])

        if os.path.exists("data/mlb_pitching_individual.csv"):
            df_temp = pd.read_csv("data/mlb_pitching_individual.csv", sep=None, engine='python', on_bad_lines='skip')
            df_temp.columns = df_temp.columns.str.strip()
            if not df_temp.empty and 'Name' in df_temp.columns:
                df_temp['xFIP'] = pd.to_numeric(df_temp['xFIP'], errors='coerce')
                df_temp['ERA'] = pd.to_numeric(df_temp['ERA'], errors='coerce')
                pitcheo_individual = df_temp.dropna(subset=['Name', 'ERA'])

        if os.path.exists("data/mlb_park_factors.csv"): 
            df_temp = pd.read_csv("data/mlb_park_factors.csv", sep=None, engine='python', on_bad_lines='skip')
            df_temp.columns = df_temp.columns.str.strip()
            if not df_temp.empty:
                park = df_temp
                
        if os.path.exists("data/mlb_games.csv"): 
            df_temp = pd.read_csv("data/mlb_games.csv", sep=None, engine='python', on_bad_lines='skip')
            df_temp.columns = df_temp.columns.str.strip()
            if not df_temp.empty:
                games = df_temp

    except Exception as e:
        st.warning(f"Aviso menor de lectura de archivos: {e}")
        
    return bateo, pitcheo, pitcheo_individual, park, games

df_bat, df_pit, df_pit_ind, df_parks, df_games = cargar_datos_historicos()

# Inicializar y entrenar el modelo de Machine Learning de manera transparente
predictor_ml = PredictorMLMLB()
if not df_games.empty and not df_bat.empty and not df_pit.empty:
    predictor_ml.entrenar(df_bat, df_pit, df_games)

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
                            "linea_carreras": None,
                            "cuota_loc": None, "cuota_vis": None, "cuota_over": None, "cuota_under": None,
                            "spread_loc": None, "cuota_spread_loc": None, "spread_vis": None, "cuota_spread_vis": None
                        }
    except Exception as e:
        st.error(f"Error en MLB StatsAPI: {e}")

    if ODDS_API_KEY != "":
        url_odds = f"https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/?apiKey={ODDS_API_KEY}&regions=us&markets=h2h,totals,spreads&oddsFormat=american"
        try:
            res_odds = requests.get(url_odds, timeout=5)
            if res_odds.status_code == 200:
                data_odds = res_odds.json()
                for item in data_odds:
                    h_team = item.get('home_team')
                    for k, p in partidos.items():
                        if h_team and p['local'] and (p['local'].lower() in h_team.lower() or h_team.lower() in p['local'].lower()):
                            bookmakers = item.get('bookmakers', [])
                            
                            h2h_encontrado = False
                            totals_encontrado = False
                            spreads_encontrado = False
                            
                            for bookmaker in bookmakers:
                                markets = bookmaker.get('markets', [])
                                for m in markets:
                                    if m['key'] == 'h2h' and not h2h_encontrado:
                                        for out in m['outcomes']:
                                            if out['name'] == h_team:
                                                p['cuota_loc'] = american_to_decimal(out['price'])
                                            else:
                                                p['cuota_vis'] = american_to_decimal(out['price'])
                                        h2h_encontrado = True
                                        
                                    elif m['key'] == 'totals' and not totals_encontrado:
                                        for out in m['outcomes']:
                                            if out['name'] == 'Over':
                                                p['linea_carreras'] = float(out['point'])
                                                p['cuota_over'] = american_to_decimal(out['price'])
                                            elif out['name'] == 'Under':
                                                p['cuota_under'] = american_to_decimal(out['price'])
                                        totals_encontrado = True

                                    elif m['key'] == 'spreads' and not spreads_encontrado:
                                        for out in m['outcomes']:
                                            if out['name'] == h_team:
                                                p['spread_loc'] = float(out['point'])
                                                p['cuota_spread_loc'] = american_to_decimal(out['price'])
                                            else:
                                                p['spread_vis'] = float(out['point'])
                                                p['cuota_spread_vis'] = american_to_decimal(out['price'])
                                        spreads_encontrado = True
                                
                                if h2h_encontrado and totals_encontrado and spreads_encontrado:
                                    break
            elif res_odds.status_code == 429:
                st.warning("⚠️ Límite de tu API Key de cuotas agotado (The-Odds-API). Pasando a modo matemático sin cuotas reales.")
            elif res_odds.status_code == 401:
                st.warning("⚠️ API Key de The-Odds-API rechazada. Revisa que sea correcta.")
            else:
                st.warning(f"⚠️ Error de servidor de cuotas: {res_odds.status_code}")
        except Exception as e:
            st.warning(f"Aviso de sincronización de cuotas: {e}")
            
    return partidos

# --- INTERFAZ ---
st.title("⚾ MLB Quant Analytics Pro (Montecarlo + Kelly)")
st.markdown("Sistema autónomo de Sabermetría, Clima en Vivo, Simulación Cuántica de Duelos, Escáner Global y Criterio de Kelly.")

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
            st.markdown("Escanea la cartelera exigiendo que **Tanto Machine Learning COMO Montecarlo** tengan >60% de probabilidad de forma individual.")
            
            if st.button("🚀 Ejecutar Escáner Global de la Jornada", type="primary"):
                with st.spinner("Escaneando duelos y procesando simulaciones avanzadas..."):
                    recomendaciones = []
                    
                    for llave, datos_partido in partidos_hoy.items():
                        loc_abbr = EQUIPOS_MAP.get(datos_partido["local"], "")
                        vis_abbr = EQUIPOS_MAP.get(datos_partido["visita"], "")
                        if not loc_abbr or not vis_abbr: continue
                        
                        cuota_loc = datos_partido.get("cuota_loc")
                        cuota_vis = datos_partido.get("cuota_vis")
                        linea_casino = datos_partido.get("linea_carreras")
                        
                        if cuota_loc is None or cuota_vis is None or linea_casino is None:
                            continue
                        
                        try:
                            team_bat_loc = df_bat[df_bat['Team'] == loc_abbr]
                            wrc_loc = float(team_bat_loc.iloc[-1]['wRC+']) if not team_bat_loc.empty else 100.0
                            
                            team_bat_vis = df_bat[df_bat['Team'] == vis_abbr]
                            wrc_vis = float(team_bat_vis.iloc[-1]['wRC+']) if not team_bat_vis.empty else 100.0
                            
                            pitcher_loc_nombre = datos_partido["pitcher_local"]
                            xfip_loc = None
                            if pitcher_loc_nombre != "Por Anunciar" and not df_pit_ind.empty:
                                match_loc = df_pit_ind[df_pit_ind['Name'].str.contains(pitcher_loc_nombre.split()[-1], case=False, na=False)]
                                if not match_loc.empty:
                                    xfip_loc = float(match_loc.iloc[-1]['xFIP'])
                            
                            if xfip_loc is None:
                                team_pit_loc = df_pit[df_pit['Team'] == loc_abbr]
                                if team_pit_loc.empty: continue
                                xfip_loc = float(team_pit_loc.iloc[-1]['xFIP'])

                            pitcher_vis_nombre = datos_partido["pitcher_visita"]
                            xfip_vis = None
                            if pitcher_vis_nombre != "Por Anunciar" and not df_pit_ind.empty:
                                match_vis = df_pit_ind[df_pit_ind['Name'].str.contains(pitcher_vis_nombre.split()[-1], case=False, na=False)]
                                if not match_vis.empty:
                                    xfip_vis = float(match_vis.iloc[-1]['xFIP'])
                            
                            if xfip_vis is None:
                                team_pit_vis = df_pit[df_pit['Team'] == vis_abbr]
                                if team_pit_vis.empty: continue
                                xfip_vis = float(team_pit_vis.iloc[-1]['xFIP'])

                            team_bullpen_loc = df_pit[df_pit['Team'] == loc_abbr]
                            bullpen_loc_era = float(team_bullpen_loc.iloc[-1]['ERA']) if not team_bullpen_loc.empty else 4.0
                            
                            team_bullpen_vis = df_pit[df_pit['Team'] == vis_abbr]
                            bullpen_vis_era = float(team_bullpen_vis.iloc[-1]['ERA']) if not team_bullpen_vis.empty else 4.0
                            
                            df_parks.columns = df_parks.columns.str.strip()
                            park_data = pd.DataFrame()

                            col_equipo_park = next((c for c in ['Team', 'TeamCode', 'Abbr', 'Franchise', 'Equipo', 'franchise'] if c in df_parks.columns), df_parks.columns[0])
                            col_pf = next((c for c in df_parks.columns if 'park_factor' in c.lower() or 'factor' in c.lower() or 'pf' in c.lower()), None)
                            col_alt = next((c for c in df_parks.columns if 'altitud' in c.lower() or 'alt' in c.lower() or 'elevation' in c.lower() or 'pie' in c.lower()), None)

                            if not col_pf or not col_alt: continue

                            park_data = df_parks[df_parks[col_equipo_park].astype(str).str.upper() == loc_abbr.upper()]

                            if park_data.empty:
                                nombre_ciudad = datos_partido["local"].split()[-1]
                                park_data = df_parks[df_parks.apply(lambda row: row.astype(str).str.contains(nombre_ciudad, case=False).any(), axis=1)]

                            if park_data.empty: continue

                            park_factor = float(park_data[col_pf].values[0])
                            altitud = float(park_data[col_alt].values[0])
                            
                            # Clima real del estadio para el scanner; fallback solo si la consulta no responde.
                            temp_scan, viento_scan, dir_scan = obtener_clima_estadio(datos_partido["local"])
                            temp_scan = 72 if temp_scan is None else temp_scan
                            viento_scan = 8 if viento_scan is None else viento_scan
                            dir_scan = "None" if not dir_scan else dir_scan

                            # Ejecutar Simulación de Montecarlo
                            res_mc = simular_partido_mlb(
                                local=datos_partido['local'], visita=datos_partido['visita'],
                                pitcher_loc_xfip=xfip_loc, pitcher_vis_xfip=xfip_vis,
                                wrc_loc=wrc_loc, wrc_vis=wrc_vis,
                                bullpen_loc_era=bullpen_loc_era, bullpen_vis_era=bullpen_vis_era,
                                park_factor=park_factor, altitud_ft=altitud,
                                viento_mph=viento_scan, direccion_viento=dir_scan, temp_f=temp_scan,
                                linea_carreras_casino=linea_casino,
                                df_games=df_games,
                                num_simulaciones=50000
                            )

                            # Ejecutar Machine Learning
                            res_ml = predictor_ml.predecir_partido(loc_abbr, vis_abbr, wrc_loc, wrc_vis, xfip_loc, xfip_vis, park_factor)

                            # --- PROBABILIDADES MONTECARLO ---
                            prob_mc_loc = res_mc['Moneyline']['Gana Local']
                            prob_mc_vis = res_mc['Moneyline']['Gana Visita']
                            carreras_dict = res_mc.get('Carreras', {})
                            prob_mc_over = carreras_dict.get(f"Over {linea_casino}", 50.0)
                            prob_mc_under = carreras_dict.get(f"Under {linea_casino}", 50.0)

                            spread_loc = datos_partido.get("spread_loc")
                            spread_vis = datos_partido.get("spread_vis")
                            
                            # Lectura dinámica de Montecarlo para el hándicap local y visitante
                            prob_mc_spread_loc = carreras_dict.get(f"Spread Local {spread_loc:+.1f}", 50.0) if spread_loc is not None else 50.0
                            prob_mc_spread_vis = carreras_dict.get(f"Spread Visita {spread_vis:+.1f}", 50.0) if spread_vis is not None else 50.0

                            # --- PROBABILIDADES MACHINE LEARNING ---
                            prob_ml_loc = res_ml['Probabilidad_Local']
                            prob_ml_vis = res_ml['Probabilidad_Visita']

                            proy_carreras = res_ml.get('Proyeccion_Carreras', linea_casino)
                            prob_ml_over = estimar_prob_ml(proy_carreras, linea_casino, "over", res_ml.get("Sigma_Carreras"))
                            prob_ml_under = estimar_prob_ml(proy_carreras, linea_casino, "under", res_ml.get("Sigma_Carreras"))

                            proy_hc_loc = res_ml.get('Proyeccion_Handicap_Local', 0)
                            prob_ml_spread_loc = estimar_prob_ml(proy_hc_loc, spread_loc, "spread_loc", res_ml.get("Sigma_Handicap")) if spread_loc is not None else 50.0
                            prob_ml_spread_vis = estimar_prob_ml(-proy_hc_loc, spread_vis, "spread_vis", res_ml.get("Sigma_Handicap")) if spread_vis is not None else 50.0

                            # --- UMBRALES DINÁMICOS POR MERCADO ---
                            umbral_ml = 55.0       # Moneyline
                            umbral_totales = 52.0  # Totales: consenso + valor relativo al mercado
                            umbral_handicap = 54.0 # Hándicap

                            # --- EVALUACIÓN Y FILTRO INTELIGENTE ---
                            # Mercado sin vig para filtrar valor real, sin cambiar las probabilidades del modelo.
                            mkt_loc_scanner, mkt_vis_scanner = _prob_no_vig_dos_vias(cuota_loc, cuota_vis)
                            mkt_over_scanner, mkt_under_scanner = _prob_no_vig_dos_vias(
                                datos_partido.get("cuota_over"), datos_partido.get("cuota_under")
                            )
                            mkt_sp_loc_scanner, mkt_sp_vis_scanner = _prob_no_vig_dos_vias(
                                datos_partido.get("cuota_spread_loc"), datos_partido.get("cuota_spread_vis")
                            )

                            
                            # 1. Moneyline Local
                            if prob_mc_loc >= umbral_ml and prob_ml_loc >= umbral_ml:
                                prob_comb_loc = (prob_mc_loc + prob_ml_loc) / 2.0
                                ev_loc = (prob_comb_loc / 100.0) * cuota_loc - 1.0
                                if _pasa_valor(prob_comb_loc, cuota_loc, mkt_loc_scanner):
                                    recomendaciones.append({
                                        "Partido": f"{datos_partido['visita']} @ {datos_partido['local']}",
                                        "Mercado": "Moneyline",
                                        "Apuesta": f"Gana Local ({datos_partido['local']})",
                                        "Prob. ML": f"{round(prob_ml_loc, 1)}%",
                                        "Prob. MC": f"{round(prob_mc_loc, 1)}%",
                                        "Cuota": cuota_loc,
                                        "EV+": f"{round(ev_loc*100, 2)}%",
                                        "Stake Kelly": f"{calcular_criterio_kelly(prob_comb_loc, cuota_loc)}%",
                                        "_Score": _score_valor(prob_comb_loc, cuota_loc, mkt_loc_scanner, abs(prob_mc_loc-prob_ml_loc))
                                    })

                            # 2. Moneyline Visita
                            if prob_mc_vis >= umbral_ml and prob_ml_vis >= umbral_ml:
                                prob_comb_vis = (prob_mc_vis + prob_ml_vis) / 2.0
                                ev_vis = (prob_comb_vis / 100.0) * cuota_vis - 1.0
                                if _pasa_valor(prob_comb_vis, cuota_vis, mkt_vis_scanner):
                                    recomendaciones.append({
                                        "Partido": f"{datos_partido['visita']} @ {datos_partido['local']}",
                                        "Mercado": "Moneyline",
                                        "Apuesta": f"Gana Visita ({datos_partido['visita']})",
                                        "Prob. ML": f"{round(prob_ml_vis, 1)}%",
                                        "Prob. MC": f"{round(prob_mc_vis, 1)}%",
                                        "Cuota": cuota_vis,
                                        "EV+": f"{round(ev_vis*100, 2)}%",
                                        "Stake Kelly": f"{calcular_criterio_kelly(prob_comb_vis, cuota_vis)}%",
                                        "_Score": _score_valor(prob_comb_vis, cuota_vis, mkt_vis_scanner, abs(prob_mc_vis-prob_ml_vis))
                                    })

                            # 3. Totales Over: consenso + edge/EV, no un 59% absoluto casi inalcanzable.
                            cuota_ov = datos_partido.get("cuota_over")
                            if cuota_ov is not None:
                                prob_comb_over = (prob_mc_over + prob_ml_over) / 2.0
                                desac_over = abs(prob_mc_over - prob_ml_over)
                                ev_over = (prob_comb_over / 100.0) * cuota_ov - 1.0
                                if (prob_mc_over >= umbral_totales and prob_ml_over >= umbral_totales and
                                    prob_comb_over >= 55.0 and desac_over <= 10.0 and
                                    _pasa_valor(prob_comb_over, cuota_ov, mkt_over_scanner, min_ev=0.04, min_edge=0.04)):
                                    recomendaciones.append({
                                        "Partido": f"{datos_partido['visita']} @ {datos_partido['local']}",
                                        "Mercado": "Totales",
                                        "Apuesta": f"Over {linea_casino}",
                                        "Prob. ML": f"{round(prob_ml_over, 1)}%",
                                        "Prob. MC": f"{round(prob_mc_over, 1)}%",
                                        "Cuota": cuota_ov,
                                        "EV+": f"{round(ev_over*100, 2)}%",
                                        "Stake Kelly": f"{calcular_criterio_kelly(prob_comb_over, cuota_ov)}%",
                                        "_Score": _score_valor(prob_comb_over, cuota_ov, mkt_over_scanner, desac_over)
                                    })

                            # 4. Totales Under: mismo estándar que Over.
                            cuota_un = datos_partido.get("cuota_under")
                            if cuota_un is not None:
                                prob_comb_under = (prob_mc_under + prob_ml_under) / 2.0
                                desac_under = abs(prob_mc_under - prob_ml_under)
                                ev_under = (prob_comb_under / 100.0) * cuota_un - 1.0
                                if (prob_mc_under >= umbral_totales and prob_ml_under >= umbral_totales and
                                    prob_comb_under >= 55.0 and desac_under <= 10.0 and
                                    _pasa_valor(prob_comb_under, cuota_un, mkt_under_scanner, min_ev=0.04, min_edge=0.04)):
                                    recomendaciones.append({
                                        "Partido": f"{datos_partido['visita']} @ {datos_partido['local']}",
                                        "Mercado": "Totales",
                                        "Apuesta": f"Under {linea_casino}",
                                        "Prob. ML": f"{round(prob_ml_under, 1)}%",
                                        "Prob. MC": f"{round(prob_mc_under, 1)}%",
                                        "Cuota": cuota_un,
                                        "EV+": f"{round(ev_under*100, 2)}%",
                                        "Stake Kelly": f"{calcular_criterio_kelly(prob_comb_under, cuota_un)}%",
                                        "_Score": _score_valor(prob_comb_under, cuota_un, mkt_under_scanner, desac_under)
                                    })
                            
                            # 5. Spread Local: consenso MC + ML; evita sesgo sistemático hacia +1.5.
                            cuota_sp_loc = datos_partido.get("cuota_spread_loc")
                            if spread_loc is not None and cuota_sp_loc is not None:
                                prob_comb_sp_loc = (prob_mc_spread_loc + prob_ml_spread_loc) / 2.0
                                desac_sp_loc = abs(prob_mc_spread_loc - prob_ml_spread_loc)
                                ev_sp_loc = (prob_comb_sp_loc / 100.0) * cuota_sp_loc - 1.0
                                
                                if (prob_mc_spread_loc >= 56.0 and prob_ml_spread_loc >= 54.0 and
                                    prob_comb_sp_loc >= 56.0 and desac_sp_loc <= 12.0 and
                                    _pasa_valor(prob_comb_sp_loc, cuota_sp_loc, mkt_sp_loc_scanner, min_ev=0.04, min_edge=0.04)):
                                    recomendaciones.append({
                                        "Partido": f"{datos_partido['visita']} @ {datos_partido['local']}",
                                        "Mercado": "Hándicap",
                                        "Apuesta": f"Hándicap {spread_loc:+.1f} ({datos_partido['local']})",
                                        "Prob. ML": f"{round(prob_ml_spread_loc, 1)}%",
                                        "Prob. MC": f"{round(prob_mc_spread_loc, 1)}%",
                                        "Cuota": cuota_sp_loc,
                                        "EV+": f"{round(ev_sp_loc*100, 2)}%",
                                        "Stake Kelly": f"{calcular_criterio_kelly(prob_comb_sp_loc, cuota_sp_loc)}%",
                                        "_Score": _score_valor(prob_comb_sp_loc, cuota_sp_loc, mkt_sp_loc_scanner, desac_sp_loc)
                                    })

                            # 6. Spread Visita: mismo estándar de consenso que el local.
                            cuota_sp_vis = datos_partido.get("cuota_spread_vis")
                            if spread_vis is not None and cuota_sp_vis is not None:
                                prob_comb_sp_vis = (prob_mc_spread_vis + prob_ml_spread_vis) / 2.0
                                desac_sp_vis = abs(prob_mc_spread_vis - prob_ml_spread_vis)
                                ev_sp_vis = (prob_comb_sp_vis / 100.0) * cuota_sp_vis - 1.0
                                
                                if (prob_mc_spread_vis >= 56.0 and prob_ml_spread_vis >= 54.0 and
                                    prob_comb_sp_vis >= 56.0 and desac_sp_vis <= 12.0 and
                                    _pasa_valor(prob_comb_sp_vis, cuota_sp_vis, mkt_sp_vis_scanner, min_ev=0.04, min_edge=0.04)):
                                    recomendaciones.append({
                                        "Partido": f"{datos_partido['visita']} @ {datos_partido['local']}",
                                        "Mercado": "Hándicap",
                                        "Apuesta": f"Hándicap {spread_vis:+.1f} ({datos_partido['visita']})",
                                        "Prob. ML": f"{round(prob_ml_spread_vis, 1)}%",
                                        "Prob. MC": f"{round(prob_mc_spread_vis, 1)}%",
                                        "Cuota": cuota_sp_vis,
                                        "EV+": f"{round(ev_sp_vis*100, 2)}%",
                                        "Stake Kelly": f"{calcular_criterio_kelly(prob_comb_sp_vis, cuota_sp_vis)}%",
                                        "_Score": _score_valor(prob_comb_sp_vis, cuota_sp_vis, mkt_sp_vis_scanner, desac_sp_vis)
                                    })

                        except Exception as e:
                            continue
                    
                    if recomendaciones:
                        df_recom = pd.DataFrame(recomendaciones)
                        if "_Score" in df_recom.columns:
                            df_recom = df_recom.sort_values("_Score", ascending=False).head(3).drop(columns=["_Score"])
                        st.dataframe(df_recom, use_container_width=True, hide_index=True)
                    else:
                        st.info("No se encontraron partidos con EV+ y más del 60% de probabilidad en ambos modelos hoy.")
        else:
            st.subheader("1. Cartelera Oficial Sincronizada")
            seleccion = st.selectbox("Selecciona un duelo:", list(partidos_hoy.keys()))
            datos_partido = partidos_hoy[seleccion]
            
            temp_auto, viento_auto, dir_auto = obtener_clima_estadio(datos_partido["local"])
            
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
                with st.spinner("Procesando datos en vivo y ejecutando simulaciones..."):
                    loc_abbr = EQUIPOS_MAP.get(datos_partido["local"], "")
                    vis_abbr = EQUIPOS_MAP.get(datos_partido["visita"], "")
                    
                    if df_bat.empty or df_pit.empty or df_parks.empty:
                        st.error("❌ Error crítico: Las bases de datos históricas están vacías.")
                        st.stop()

                    try:
                        team_bat_loc = df_bat[df_bat['Team'] == loc_abbr]
                        wrc_loc = float(team_bat_loc.iloc[-1]['wRC+']) if not team_bat_loc.empty else 100.0
                        
                        team_bat_vis = df_bat[df_bat['Team'] == vis_abbr]
                        wrc_vis = float(team_bat_vis.iloc[-1]['wRC+']) if not team_bat_vis.empty else 100.0
                    except Exception as e:
                        st.error(f"Error procesando wRC+ de bateo: {e}")
                        st.stop()
                    
                    pitcher_loc_nombre = datos_partido["pitcher_local"]
                    xfip_loc = None
                    if pitcher_loc_nombre != "Por Anunciar" and not df_pit_ind.empty:
                        match_loc = df_pit_ind[df_pit_ind['Name'].str.contains(pitcher_loc_nombre.split()[-1], case=False, na=False)]
                        if not match_loc.empty:
                            xfip_loc = float(match_loc.iloc[-1]['xFIP']) 
                    
                    if xfip_loc is None:
                        team_pit_loc = df_pit[df_pit['Team'] == loc_abbr]
                        if team_pit_loc.empty:
                            st.error("❌ No hay datos de pitcheo reales para el local.")
                            st.stop()
                        xfip_loc = float(team_pit_loc.iloc[-1]['xFIP']) 

                    pitcher_vis_nombre = datos_partido["pitcher_visita"]
                    xfip_vis = None
                    if pitcher_vis_nombre != "Por Anunciar" and not df_pit_ind.empty:
                        match_vis = df_pit_ind[df_pit_ind['Name'].str.contains(pitcher_vis_nombre.split()[-1], case=False, na=False)]
                        if not match_vis.empty:
                            xfip_vis = float(match_vis.iloc[-1]['xFIP']) 
                    
                    if xfip_vis is None:
                        team_pit_vis = df_pit[df_pit['Team'] == vis_abbr]
                        if team_pit_vis.empty:
                            st.error("❌ No hay datos de pitcheo reales para el visitante.")
                            st.stop()
                        xfip_vis = float(team_pit_vis.iloc[-1]['xFIP']) 

                    team_bullpen_loc = df_pit[df_pit['Team'] == loc_abbr]
                    bullpen_loc_era = float(team_bullpen_loc.iloc[-1]['ERA']) if not team_bullpen_loc.empty else 4.0
                    
                    team_bullpen_vis = df_pit[df_pit['Team'] == vis_abbr]
                    bullpen_vis_era = float(team_bullpen_vis.iloc[-1]['ERA']) if not team_bullpen_vis.empty else 4.0
                    
                    df_parks.columns = df_parks.columns.str.strip()
                    park_data = pd.DataFrame()

                    col_equipo_park = next((c for c in ['Team', 'TeamCode', 'Abbr', 'Franchise', 'Equipo', 'franchise'] if c in df_parks.columns), df_parks.columns[0])
                    col_pf = next((c for c in df_parks.columns if 'park_factor' in c.lower() or 'factor' in c.lower() or 'pf' in c.lower()), None)
                    col_alt = next((c for c in df_parks.columns if 'altitud' in c.lower() or 'alt' in c.lower() or 'elevation' in c.lower() or 'pie' in c.lower()), None)

                    if not col_pf or not col_alt:
                        st.error("❌ El archivo `mlb_park_factors.csv` no contiene columnas reconocibles de Park Factor o Altitud.")
                        st.stop()

                    park_data = df_parks[df_parks[col_equipo_park].astype(str).str.upper() == loc_abbr.upper()]

                    if park_data.empty:
                        nombre_ciudad = datos_partido["local"].split()[-1]
                        park_data = df_parks[df_parks.apply(lambda row: row.astype(str).str.contains(nombre_ciudad, case=False).any(), axis=1)]

                    if park_data.empty:
                        st.error(f"❌ Error de integridad: No se encontró ningún registro real para el equipo '{datos_partido['local']}' ({loc_abbr}) en 'mlb_park_factors.csv'. Verifica tu archivo de estadios.")
                        st.stop()

                    park_factor = float(park_data[col_pf].values[0])
                    altitud = float(park_data[col_alt].values[0])
                    
                    linea_casino = datos_partido["linea_carreras"]
                    
                    if linea_casino is None or datos_partido["cuota_loc"] is None or datos_partido["cuota_vis"] is None:
                         st.error("❌ Faltan cuotas reales del casino. Simulación cancelada para no inyectar datos falsos.")
                         st.stop()

                    res_mc = simular_partido_mlb(
                        local=datos_partido['local'], visita=datos_partido['visita'],
                        pitcher_loc_xfip=xfip_loc, pitcher_vis_xfip=xfip_vis,
                        wrc_loc=wrc_loc, wrc_vis=wrc_vis,
                        bullpen_loc_era=bullpen_loc_era, bullpen_vis_era=bullpen_vis_era,
                        park_factor=park_factor, altitud_ft=altitud,
                        viento_mph=viento, direccion_viento=dir_viento, temp_f=temp,
                        linea_carreras_casino=linea_casino,
                        df_games=df_games,
                        num_simulaciones=50000
                    )
                    
                    cuotas_reales = {
                        "Moneyline_Local": datos_partido["cuota_loc"],
                        "Moneyline_Visita": datos_partido["cuota_vis"],
                        "Cuota_Over": datos_partido["cuota_over"],
                        "Cuota_Under": datos_partido["cuota_under"],
                        "Spread_Local": datos_partido.get("spread_loc"),
                        "Cuota_Spread_Local": datos_partido.get("cuota_spread_loc"),
                        "Spread_Visita": datos_partido.get("spread_vis"),
                        "Cuota_Spread_Visita": datos_partido.get("cuota_spread_vis")
                    }
                    
                    veredicto_apuestas = []
                    prob_loc = res_mc['Moneyline']['Gana Local']
                    prob_vis = res_mc['Moneyline']['Gana Visita']
                    cuota_loc = cuotas_reales['Moneyline_Local']
                    cuota_vis = cuotas_reales['Moneyline_Visita']
                    
                    # Moneyline Local
                    kelly_loc = calcular_criterio_kelly(prob_loc, cuota_loc)
                    ev_loc = (prob_loc / 100.0) * cuota_loc - 1.0
                    veredicto_apuestas.append({
                        "Apuesta": f"Gana Local ({datos_partido['local']})",
                        "Prob. Real": f"{prob_loc}%",
                        "Cuota": cuota_loc,
                        "EV+": f"{round(ev_loc*100, 2)}%",
                        "Kelly Stake": f"{kelly_loc}%"
                    })
                    
                    # Moneyline Visita
                    kelly_vis = calcular_criterio_kelly(prob_vis, cuota_vis)
                    ev_vis = (prob_vis / 100.0) * cuota_vis - 1.0
                    veredicto_apuestas.append({
                        "Apuesta": f"Gana Visita ({datos_partido['visita']})",
                        "Prob. Real": f"{prob_vis}%",
                        "Cuota": cuota_vis,
                        "EV+": f"{round(ev_vis*100, 2)}%",
                        "Kelly Stake": f"{kelly_vis}%"
                    })

                    # Over / Under (Totales)
                    carreras_dict = res_mc.get('Carreras', {})
                    prob_over = carreras_dict.get(f"Over {linea_casino}", 50.0)
                    cuota_ov = cuotas_reales.get("Cuota_Over")
                    if cuota_ov is not None:
                        kelly_ov = calcular_criterio_kelly(prob_over, cuota_ov)
                        ev_ov = (prob_over / 100.0) * cuota_ov - 1.0
                        veredicto_apuestas.append({
                            "Apuesta": f"Over {linea_casino}",
                            "Prob. Real": f"{prob_over}%",
                            "Cuota": cuota_ov,
                            "EV+": f"{round(ev_ov*100, 2)}%",
                            "Kelly Stake": f"{kelly_ov}%"
                        })

                    prob_under = carreras_dict.get(f"Under {linea_casino}", 50.0)
                    cuota_un = cuotas_reales.get("Cuota_Under")
                    if cuota_un is not None:
                        kelly_un = calcular_criterio_kelly(prob_under, cuota_un)
                        ev_un = (prob_under / 100.0) * cuota_un - 1.0
                        veredicto_apuestas.append({
                            "Apuesta": f"Under {linea_casino}",
                            "Prob. Real": f"{prob_under}%",
                            "Cuota": cuota_un,
                            "EV+": f"{round(ev_un*100, 2)}%",
                            "Kelly Stake": f"{kelly_un}%"
                        })

                    # Hándicap (Spreads reales desde Montecarlo con lectura dinámica)
                    spread_loc = cuotas_reales.get("Spread_Local")
                    cuota_sp_loc = cuotas_reales.get("Cuota_Spread_Local")
                    if spread_loc is not None and cuota_sp_loc is not None:
                        prob_spread_loc = carreras_dict.get(f"Spread Local {spread_loc:+.1f}", prob_loc * 0.90)
                        kelly_sp_loc = calcular_criterio_kelly(prob_spread_loc, cuota_sp_loc)
                        ev_sp_loc = (prob_spread_loc / 100.0) * cuota_sp_loc - 1.0
                        veredicto_apuestas.append({
                            "Apuesta": f"Hándicap {spread_loc:+.1f} ({datos_partido['local']})",
                            "Prob. Real": f"{round(prob_spread_loc, 2)}%",
                            "Cuota": cuota_sp_loc,
                            "EV+": f"{round(ev_sp_loc*100, 2)}%",
                            "Kelly Stake": f"{kelly_sp_loc}%"
                        })

                    spread_vis = cuotas_reales.get("Spread_Visita")
                    cuota_sp_vis = cuotas_reales.get("Cuota_Spread_Visita")
                    if spread_vis is not None and cuota_sp_vis is not None:
                        prob_spread_vis = carreras_dict.get(f"Spread Visita {spread_vis:+.1f}", prob_vis * 0.90)
                        kelly_sp_vis = calcular_criterio_kelly(prob_spread_vis, cuota_sp_vis)
                        ev_sp_vis = (prob_spread_vis / 100.0) * cuota_sp_vis - 1.0
                        veredicto_apuestas.append({
                            "Apuesta": f"Hándicap {spread_vis:+.1f} ({datos_partido['visita']})",
                            "Prob. Real": f"{round(prob_spread_vis, 2)}%",
                            "Cuota": cuota_sp_vis,
                            "EV+": f"{round(ev_sp_vis*100, 2)}%",
                            "Kelly Stake": f"{kelly_sp_vis}%"
                        })
                    
                    df_apuestas = pd.DataFrame(veredicto_apuestas)
                    
                    st.markdown("---")
                    st.subheader(f"🏟️ Factores Ambientales en {datos_partido['local']}")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Altitud del Parque", f"{altitud} ft")
                    m2.metric("Park Factor General", park_factor)
                    m3.metric("Clima in Vivo", f"{temp}°F | Viento: {viento}mph")
                    
                    st.markdown("### 🎲 Probabilidades Reales del Duelo")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric(f"Gana {datos_partido['local']}", f"{res_mc['Moneyline']['Gana Local']}%")
                    c2.metric(f"Gana {datos_partido['visita']}", f"{res_mc['Moneyline']['Gana Visita']}%")
                    
                    prob_over = res_mc.get('Carreras', {}).get(f"Over {linea_casino}", 50.0)
                    c3.metric(f"Over {linea_casino} Carreras", f"{prob_over}%")
                    c4.metric("Promedio Carreras Total", f"{res_mc['Carreras']['Promedio_Total']}")
                    
                    st.markdown("### 🎯 Veredicto Financiero y Valor Esperado (EV+)")
                    st.dataframe(df_apuestas, use_container_width=True, hide_index=True)
