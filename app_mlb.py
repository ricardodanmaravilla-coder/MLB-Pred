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
            st.markdown("Analiza automáticamente todos los duelos del día filtrando oportunidades donde **tanto Machine Learning como Montecarlo superan el 60% de probabilidad** y calcula la asignación de capital mediante el **Criterio de Kelly Fraccionado**.")
            
            if st.button("🚀 Ejecutar Escáner Global de la Jornada", type="primary"):
                with st.spinner("Escaneando todos los encuentros, procesando modelos predictivos y aplicando Kelly..."):
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
                            
                            preds_ml = ml.predecir_partido(loc_abbr, vis_abbr, wrc_loc, wrc_vis, xfip_loc, xfip_vis, park_factor)
                            res_mc = simular_partido_mlb(
                                local=datos_partido['local'], visita=datos_partido['visita'],
                                pitcher_loc_xfip=xfip_loc, pitcher_vis_xfip=xfip_vis,
                                wrc_loc=wrc_loc, wrc_vis=wrc_vis,
                                bullpen_loc_era=bullpen_loc_era, bullpen_vis_era=bullpen_vis_era,
                                park_factor=park_factor, altitud_ft=altitud,
                                viento_mph=8, direccion_viento="None", temp_f=72,
                                linea_carreras_casino=linea_casino, num_simulaciones=100000
                            )
                            
                            prob_ml_loc = preds_ml.get('Probabilidad_Local', 50.0)
                            prob_ml_vis = preds_ml.get('Probabilidad_Visita', 50.0)
                            prob_mc_loc = res_mc['Moneyline']['Gana Local']
                            prob_mc_vis = res_mc['Moneyline']['Gana Visita']
                            
                            # Filtro estricto: Ambos modelos (ML y Montecarlo) deben dar > 60%
                            if prob_ml_loc >= 60.0 and prob_mc_loc >= 60.0:
                                cuota = datos_partido["cuota_loc"]
                                kelly_pct = calcular_criterio_kelly(prob_mc_loc, cuota)
                                recomendaciones.append({
                                    "Partido": f"{datos_partido['visita']} @ {datos_partido['local']}",
                                    "Apuesta Sugerida": f"Gana Local ({datos_partido['local']})",
                                    "Prob. ML": f"{prob_ml_loc}%",
                                    "Prob. Montecarlo": f"{prob_mc_loc}%",
                                    "Cuota Decimal": cuota,
                                    "Stake Kelly (%)": f"{kelly_pct}% del Bankroll"
                                })
                            elif prob_ml_vis >= 60.0 and prob_mc_vis >= 60.0:
                                cuota = datos_partido["cuota_vis"]
                                kelly_pct = calcular_criterio_kelly(prob_mc_vis, cuota)
                                recomendaciones.append({
                                    "Partido": f"{datos_partido['visita']} @ {datos_partido['local']}",
                                    "Apuesta Sugerida": f"Gana Visita ({datos_partido['visita']})",
                                    "Prob. ML": f"{prob_ml_vis}%",
                                    "Prob. Montecarlo": f"{prob_mc_vis}%",
                                    "Cuota Decimal": cuota,
                                    "Stake Kelly (%)": f"{kelly_pct}% del Bankroll"
                                })
                        except Exception:
                            continue
                    
                    if recomendaciones:
                        st.success(f"🎯 ¡Se encontraron {len(recomendaciones)} oportunidades de alta confluencia (>60%) en la jornada!")
                        df_recom = pd.DataFrame(recomendaciones)
                        st.dataframe(df_recom, use_container_width=True, hide_index=True)
                    else:
                        st.info("ℹ️ No hay partidos en la cartelera actual donde tanto Machine Learning como Montecarlo superen el umbral estricto del 60%. El mercado está muy parejo hoy.")
        
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
                with st.spinner("Procesando datos en vivo y ejecutando 500,000 escenarios..."):
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
                    
                    col_nombre_pitcher = 'Name'
                    for posible_col in ['Name', 'PlayerName', 'jugador', 'pitcher']:
                        if posible_col in df_pit.columns:
                            col_nombre_pitcher = posible_col
                            break

                    pitcher_loc_nombre = datos_partido["pitcher_local"]
                    xfip_loc = None
                    if pitcher_loc_nombre != "Por Anunciar" and col_nombre_pitcher in df_pit.columns:
                        match_loc = df_pit[df_pit[col_nombre_pitcher].str.contains(pitcher_loc_nombre.split()[-1], case=False, na=False)]
                        if not match_loc.empty:
                            xfip_loc = float(match_loc['xFIP'].values[0])
                    
                    if xfip_loc is None:
                        team_pit_loc = df_pit[df_pit['Team'] == loc_abbr]
                        xfip_loc = float(team_pit_loc['xFIP'].mean())

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
                    
                    df_parks.columns = df_parks.columns.str.strip()
                    park_data = df_parks[df_parks['Team'] == loc_abbr]
                    if park_data.empty:
                        park_data = df_parks[df_parks.apply(lambda row: row.astype(str).str.contains(datos_partido["local"].split()[-1], case=False).any(), axis=1)]
                    
                    if park_data.empty:
                        st.error(f"❌ No se encontró el registro para el equipo '{loc_abbr}' en el archivo de factores de estadios.")
                        st.stop()

                    col_pf = [c for c in park_data.columns if 'park_factor' in c.lower() or 'factor' in c.lower()][0]
                    col_alt = [c for c in park_data.columns if 'altitud' in c.lower() or 'alt' in c.lower()][0]

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
