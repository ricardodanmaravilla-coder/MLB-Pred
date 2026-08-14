import streamlit as st
import pandas as pd
import requests
import numpy as np
import os
import datetime

from modules.montecarlo_mlb import simular_partido_mlb
from modules.odds_mlb import analizar_apuestas_mlb
from modules.ml_mlb import PredictorMLMLB

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="MLB Quant Analytics", layout="wide", page_icon="⚾")

ODDS_API_KEY = "4725b4a69b90b1310a23134c58f3de9c"

EQUIPOS_MAP = {
    "New York Yankees": "NYY", "Boston Red Sox": "BOS", "Los Angeles Dodgers": "LAD",
    "Houston Astros": "HOU", "Atlanta Braves": "ATL", "Philadelphia Phillies": "PHI",
    "Baltimore Orioles": "BAL", "Tampa Bay Rays": "TB", "Toronto Blue Jays": "TOR",
    "Chicago White Sox": "CWS", "Cleveland Guardians": "CLE", "Detroit Tigers": "DET",
    "Kansas City Royals": "KC", "Minnesota Twins": "MIN", "Los Angeles Angels": "LAA",
    "Oakland Athletics": "OAK", "Seattle Mariners": "SEA", "Texas Rangers": "TEX",
    "Chicago Cubs": "CHC", "Cincinnati Reds": "CIN", "Milwaukee Brewers": "MIL",
    "Pittsburgh Pirates": "PIT", "St. Louis Cardinals": "STL", "Arizona Diamondbacks": "AZ",
    "Colorado Rockies": "COL", "San Francisco Giants": "SF", "San Diego Padres": "SD",
    "Miami Marlins": "MIA", "New York Mets": "NYM", "Washington Nationals": "WSH"
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

        # Carga del nuevo archivo dedicado de abridores individuales con nombres reales
        if os.path.exists("data/mlb_pitching_individual.csv"):
            df_temp = pd.read_csv("data/mlb_pitching_individual.csv", sep=None, engine='python', on_bad_lines='skip')
            df_temp.columns = df_temp.columns.str.strip()
            if not df_temp.empty and 'Name' in df_temp.columns:
                df_temp['xFIP'] = pd.to_numeric(df_temp['xFIP'], errors='coerce')
                df_temp['ERA'] = pd.to_numeric(df_temp['ERA'], errors='coerce')
                pitcheo_individual = df_temp.dropna(subset=['Name', 'xFIP'])

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
            st.markdown("Escanea toda la cartelera cruzando **Montecarlo, Machine Learning y Cuotas en Vivo** (Exigiendo >60% de probabilidad real y EV+).")
            
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
                            wrc_loc = float(df_bat[df_bat['Team'] == loc_abbr]['wRC+'].mean())
                            wrc_vis = float(df_bat[df_bat['Team'] == vis_abbr]['wRC+'].mean())
                            
                            # Lectura de xFIP del pitcher local (Plan A: Individual, Plan B: Equipo)
                            pitcher_loc_nombre = datos_partido["pitcher_local"]
                            xfip_loc = None
                            if pitcher_loc_nombre != "Por Anunciar" and not df_pit_ind.empty:
                                match_loc = df_pit_ind[df_pit_ind['Name'].str.contains(pitcher_loc_nombre.split()[-1], case=False, na=False)]
                                if not match_loc.empty:
                                    xfip_loc = float(match_loc['xFIP'].values[0])
                            
                            if xfip_loc is None:
                                team_pit_loc = df_pit[df_pit['Team'] == loc_abbr]
                                if team_pit_loc.empty: continue
                                xfip_loc = float(team_pit_loc['xFIP'].mean())

                            # Lectura de xFIP del pitcher visitante (Plan A: Individual, Plan B: Equipo)
                            pitcher_vis_nombre = datos_partido["pitcher_visita"]
                            xfip_vis = None
                            if pitcher_vis_nombre != "Por Anunciar" and not df_pit_ind.empty:
                                match_vis = df_pit_ind[df_pit_ind['Name'].str.contains(pitcher_vis_nombre.split()[-1], case=False, na=False)]
                                if not match_vis.empty:
                                    xfip_vis = float(match_vis['xFIP'].values[0])
                            
                            if xfip_vis is None:
                                team_pit_vis = df_pit[df_pit['Team'] == vis_abbr]
                                if team_pit_vis.empty: continue
                                xfip_vis = float(team_pit_vis['xFIP'].mean())

                            bullpen_loc_era = float(df_pit[df_pit['Team'] == loc_abbr]['ERA'].mean())
                            bullpen_vis_era = float(df_pit[df_pit['Team'] == vis_abbr]['ERA'].mean())
                            
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
                            
                            # Ejecutar Simulación de Montecarlo
                            res_mc = simular_partido_mlb(
                                local=datos_partido['local'], visita=datos_partido['visita'],
                                pitcher_loc_xfip=xfip_loc, pitcher_vis_xfip=xfip_vis,
                                wrc_loc=wrc_loc, wrc_vis=wrc_vis,
                                bullpen_loc_era=bullpen_loc_era, bullpen_vis_era=bullpen_vis_era,
                                park_factor=park_factor, altitud_ft=altitud,
                                viento_mph=8, direccion_viento="None", temp_f=72,
                                linea_carreras_casino=linea_casino,
                                df_games=df_games,
                                num_simulaciones=1000000
                            )

                            # Ejecutar Machine Learning
                            res_ml = predictor_ml.predecir_partido(loc_abbr, vis_abbr, wrc_loc, wrc_vis, xfip_loc, xfip_vis, park_factor)

                            # REGLA ESTRICTA: Combinación Montecarlo + ML con mínimo 60% de probabilidad
                            prob_mc_loc = res_mc['Moneyline']['Gana Local']
                            prob_mc_vis = res_mc['Moneyline']['Gana Visita']
                            prob_ml_loc = res_ml['Probabilidad_Local']
                            prob_ml_vis = res_ml['Probabilidad_Visita']

                            prob_comb_loc = (prob_mc_loc + prob_ml_loc) / 2.0
                            prob_comb_vis = (prob_mc_vis + prob_ml_vis) / 2.0

                            umbral_apuesta = 60.0 # Exigencia superior al 60%

                            # 1. Moneyline Local
                            ev_loc = (prob_comb_loc / 100.0) * cuota_loc - 1.0
                            if prob_comb_loc >= umbral_apuesta and ev_loc > 0:
                                kelly_pct = calcular_criterio_kelly(prob_comb_loc, cuota_loc)
                                recomendaciones.append({
                                    "Partido": f"{datos_partido['visita']} @ {datos_partido['local']}",
                                    "Mercado": "Moneyline",
                                    "Apuesta": f"Gana Local ({datos_partido['local']})",
                                    "Prob. Real": f"{round(prob_comb_loc, 2)}%",
                                    "Cuota": cuota_loc,
                                    "EV+": f"{round(ev_loc*100, 2)}%",
                                    "Stake Kelly": f"{kelly_pct}%"
                                })
                                
                            # 2. Moneyline Visita
                            ev_vis = (prob_comb_vis / 100.0) * cuota_vis - 1.0
                            if prob_comb_vis >= umbral_apuesta and ev_vis > 0:
                                kelly_pct = calcular_criterio_kelly(prob_comb_vis, cuota_vis)
                                recomendaciones.append({
                                    "Partido": f"{datos_partido['visita']} @ {datos_partido['local']}",
                                    "Mercado": "Moneyline",
                                    "Apuesta": f"Gana Visita ({datos_partido['visita']})",
                                    "Prob. Real": f"{round(prob_comb_vis, 2)}%",
                                    "Cuota": cuota_vis,
                                    "EV+": f"{round(ev_vis*100, 2)}%",
                                    "Stake Kelly": f"{kelly_pct}%"
                                })
                                
                            # 3. Totales (Over / Under)
                            carreras_dict = res_mc.get('Carreras', {})
                            prob_over = carreras_dict.get(f"Over {linea_casino}", 50.0)
                            cuota_ov = datos_partido.get("cuota_over")
                            if cuota_ov is not None:
                                ev_over = (prob_over / 100.0) * cuota_ov - 1.0
                                if prob_over >= umbral_apuesta and ev_over > 0:
                                    recomendaciones.append({
                                        "Partido": f"{datos_partido['visita']} @ {datos_partido['local']}",
                                        "Mercado": "Totales",
                                        "Apuesta": f"Over {linea_casino}",
                                        "Prob. Real": f"{prob_over}%",
                                        "Cuota": cuota_ov,
                                        "EV+": f"{round(ev_over*100, 2)}%",
                                        "Stake Kelly": f"{calcular_criterio_kelly(prob_over, cuota_ov)}%"
                                    })
                                
                            prob_under = carreras_dict.get(f"Under {linea_casino}", 50.0)
                            cuota_un = datos_partido.get("cuota_under")
                            if cuota_un is not None:
                                ev_under = (prob_under / 100.0) * cuota_un - 1.0
                                if prob_under >= umbral_apuesta and ev_under > 0:
                                    recomendaciones.append({
                                        "Partido": f"{datos_partido['visita']} @ {datos_partido['local']}",
                                        "Mercado": "Totales",
                                        "Apuesta": f"Under {linea_casino}",
                                        "Prob. Real": f"{prob_under}%",
                                        "Cuota": cuota_un,
                                        "EV+": f"{round(ev_under*100, 2)}%",
                                        "Stake Kelly": f"{calcular_criterio_kelly(prob_under, cuota_un)}%"
                                    })

                            # 4. Hándicap (Spreads)
                            spread_loc = datos_partido.get("spread_loc")
                            cuota_sp_loc = datos_partido.get("cuota_spread_loc")
                            if spread_loc is not None and cuota_sp_loc is not None:
                                prob_sp_loc = prob_comb_loc * 0.95 if spread_loc < 0 else prob_comb_loc * 1.05
                                prob_sp_loc = np.clip(prob_sp_loc, 1.0, 99.0)
                                ev_sp_loc = (prob_sp_loc / 100.0) * cuota_sp_loc - 1.0
                                if prob_sp_loc >= umbral_apuesta and ev_sp_loc > 0:
                                    recomendaciones.append({
                                        "Partido": f"{datos_partido['visita']} @ {datos_partido['local']}",
                                        "Mercado": "Hándicap",
                                        "Apuesta": f"Hándicap {spread_loc} ({datos_partido['local']})",
                                        "Prob. Real": f"{round(prob_sp_loc, 2)}%",
                                        "Cuota": cuota_sp_loc,
                                        "EV+": f"{round(ev_sp_loc*100, 2)}%",
                                        "Stake Kelly": f"{calcular_criterio_kelly(prob_sp_loc, cuota_sp_loc)}%"
                                    })

                            spread_vis = datos_partido.get("spread_vis")
                            cuota_sp_vis = datos_partido.get("cuota_spread_vis")
                            if spread_vis is not None and cuota_sp_vis is not None:
                                prob_sp_vis = prob_comb_vis * 0.95 if spread_vis < 0 else prob_comb_vis * 1.05
                                prob_sp_vis = np.clip(prob_sp_vis, 1.0, 99.0)
                                ev_sp_vis = (prob_sp_vis / 100.0) * cuota_sp_vis - 1.0
                                if prob_sp_vis >= umbral_apuesta and ev_sp_vis > 0:
                                    recomendaciones.append({
                                        "Partido": f"{datos_partido['visita']} @ {datos_partido['local']}",
                                        "Mercado": "Hándicap",
                                        "Apuesta": f"Hándicap {spread_vis} ({datos_partido['visita']})",
                                        "Prob. Real": f"{round(prob_sp_vis, 2)}%",
                                        "Cuota": cuota_sp_vis,
                                        "EV+": f"{round(ev_sp_vis*100, 2)}%",
                                        "Stake Kelly": f"{calcular_criterio_kelly(prob_sp_vis, cuota_sp_vis)}%"
                                    })

                        except Exception as e:
                            continue
                    
                    if recomendaciones:
                        df_recom = pd.DataFrame(recomendaciones)
                        st.dataframe(df_recom, use_container_width=True, hide_index=True)
                    else:
                        st.info("No se encontraron partidos con EV+ y más del 60% de probabilidad real hoy.")
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
                        wrc_loc = float(df_bat[df_bat['Team'] == loc_abbr]['wRC+'].mean())
                        wrc_vis = float(df_bat[df_bat['Team'] == vis_abbr]['wRC+'].mean())
                    except Exception as e:
                        st.error(f"Error procesando wRC+ de bateo: {e}")
                        st.stop()
                    
                    # Búsqueda individual de abridor local (Plan A: Individual, Plan B: Equipo)
                    pitcher_loc_nombre = datos_partido["pitcher_local"]
                    xfip_loc = None
                    if pitcher_loc_nombre != "Por Anunciar" and not df_pit_ind.empty:
                        match_loc = df_pit_ind[df_pit_ind['Name'].str.contains(pitcher_loc_nombre.split()[-1], case=False, na=False)]
                        if not match_loc.empty:
                            xfip_loc = float(match_loc['xFIP'].values[0])
                    
                    if xfip_loc is None:
                        team_pit_loc = df_pit[df_pit['Team'] == loc_abbr]
                        if team_pit_loc.empty:
                            st.error("❌ No hay datos de pitcheo reales para el local.")
                            st.stop()
                        xfip_loc = float(team_pit_loc['xFIP'].mean())

                    # Búsqueda individual de abridor visitante (Plan A: Individual, Plan B: Equipo)
                    pitcher_vis_nombre = datos_partido["pitcher_visita"]
                    xfip_vis = None
                    if pitcher_vis_nombre != "Por Anunciar" and not df_pit_ind.empty:
                        match_vis = df_pit_ind[df_pit_ind['Name'].str.contains(pitcher_vis_nombre.split()[-1], case=False, na=False)]
                        if not match_vis.empty:
                            xfip_vis = float(match_vis['xFIP'].values[0])
                    
                    if xfip_vis is None:
                        team_pit_vis = df_pit[df_pit['Team'] == vis_abbr]
                        if team_pit_vis.empty:
                            st.error("❌ No hay datos de pitcheo reales para el visitante.")
                            st.stop()
                        xfip_vis = float(team_pit_vis['xFIP'].mean())

                    bullpen_loc_era = float(df_pit[df_pit['Team'] == loc_abbr]['ERA'].mean())
                    bullpen_vis_era = float(df_pit[df_pit['Team'] == vis_abbr]['ERA'].mean())
                    
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
                        num_simulaciones=500000
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

                    # Hándicap (Spreads)
                    spread_loc = cuotas_reales.get("Spread_Local")
                    cuota_sp_loc = cuotas_reales.get("Cuota_Spread_Local")
                    if spread_loc is not None and cuota_sp_loc is not None:
                        prob_spread_loc = prob_loc * 0.95 if spread_loc < 0 else prob_loc * 1.05
                        prob_spread_loc = np.clip(prob_spread_loc, 1.0, 99.0)
                        kelly_sp_loc = calcular_criterio_kelly(prob_spread_loc, cuota_sp_loc)
                        ev_sp_loc = (prob_spread_loc / 100.0) * cuota_sp_loc - 1.0
                        veredicto_apuestas.append({
                            "Apuesta": f"Hándicap {spread_loc} ({datos_partido['local']})",
                            "Prob. Real": f"{round(prob_spread_loc, 2)}%",
                            "Cuota": cuota_sp_loc,
                            "EV+": f"{round(ev_sp_loc*100, 2)}%",
                            "Kelly Stake": f"{kelly_sp_loc}%"
                        })

                    spread_vis = cuotas_reales.get("Spread_Visita")
                    cuota_sp_vis = cuotas_reales.get("Cuota_Spread_Visita")
                    if spread_vis is not None and cuota_sp_vis is not None:
                        prob_spread_vis = prob_vis * 0.95 if spread_vis < 0 else prob_vis * 1.05
                        prob_spread_vis = np.clip(prob_spread_vis, 1.0, 99.0)
                        kelly_sp_vis = calcular_criterio_kelly(prob_spread_vis, cuota_sp_vis)
                        ev_sp_vis = (prob_spread_vis / 100.0) * cuota_sp_vis - 1.0
                        veredicto_apuestas.append({
                            "Apuesta": f"Hándicap {spread_vis} ({datos_partido['visita']})",
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
