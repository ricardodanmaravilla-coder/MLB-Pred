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
from modules.scanner_engine import (
    no_vig_two_way, moneyline_candidate, total_candidate, runline_candidate
)
from modules.pick_ledger import append_snapshot

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


def _diagnostico_total(prob_ml, prob_mc, prob_comb, cuota, mercado_no_vig, desacuerdo):
    try:
        p=float(prob_comb)/100.0
        ev=(p*float(cuota)-1.0)*100.0 if cuota is not None else None
        edge=(p-float(mercado_no_vig))*100.0 if mercado_no_vig is not None else None
        checks=[
            (float(prob_ml)>=52.0, f"ML {float(prob_ml):.1f}% < 52%"),
            (float(prob_mc)>=52.0, f"MC {float(prob_mc):.1f}% < 52%"),
            (float(prob_comb)>=54.0, f"Combinada {float(prob_comb):.1f}% < 54%"),
            (float(desacuerdo)<=15.0, f"Desacuerdo {float(desacuerdo):.1f} pp > 15"),
            (edge is not None and edge>=4.0, "Edge < 4 pp"),
            (ev is not None and ev>=4.0, "EV < 4%"),
        ]
        fails=[msg for ok,msg in checks if not ok]
        return {"EV_pct": None if ev is None else round(ev,2), "Edge_pp": None if edge is None else round(edge,2), "Estado": "CANDIDATO" if not fails else "NO BET", "Motivo": "Cumple filtros O/U" if not fails else "; ".join(fails)}
    except Exception as e:
        return {"EV_pct":None,"Edge_pp":None,"Estado":"NO BET","Motivo":f"Error diagnóstico: {e}"}


def _team_prior_stat(df, team, column, fallback):
    try:
        if df is None or df.empty or column not in df.columns or 'Team' not in df.columns:
            return float(fallback)
        x = df[df['Team'] == team].copy()
        if x.empty:
            return float(fallback)
        if 'Season' in x.columns:
            x['_SeasonNum'] = pd.to_numeric(x['Season'], errors='coerce')
            target = datetime.date.today().year - 1
            eligible = x[x['_SeasonNum'] <= target].sort_values('_SeasonNum')
            if not eligible.empty:
                val = pd.to_numeric(eligible.iloc[-1][column], errors='coerce')
                if pd.notna(val):
                    return float(val)
        val = pd.to_numeric(x.iloc[-1][column], errors='coerce')
        return float(val) if pd.notna(val) else float(fallback)
    except Exception:
        return float(fallback)


def _current_offensive_index(df, team, fallback=100.0):
    """Return current team offense centered at league average = 100.

    The repository legacy wRC+ column is OPS*100, not real wRC+. Monte Carlo
    needs a centered multiplier, so normalize the latest-season OPS index by
    that season's league median before passing it to the simulator.
    """
    try:
        if df is None or df.empty or 'Team' not in df.columns:
            return float(fallback)
        col = 'OPS_Index' if 'OPS_Index' in df.columns else ('wRC+' if 'wRC+' in df.columns else None)
        if col is None:
            return float(fallback)
        x=df.copy(); x['_v']=pd.to_numeric(x[col],errors='coerce')
        if 'Season' in x.columns:
            x['_s']=pd.to_numeric(x['Season'],errors='coerce')
            latest=x['_s'].dropna().max()
            season=x[x['_s']==latest]
        else:
            season=x
        team_rows=season[season['Team']==team]
        if team_rows.empty:
            team_rows=x[x['Team']==team]
        if team_rows.empty:
            return float(fallback)
        team_val=float(team_rows['_v'].dropna().iloc[-1])
        league_vals=season['_v'].dropna()
        center=float(league_vals.median()) if len(league_vals) else float(x['_v'].dropna().median())
        if not center or pd.isna(center):
            return float(fallback)
        return float(np.clip((team_val/center)*100.0,75.0,125.0))
    except Exception:
        return float(fallback)


def _starter_run_prevention(df, pitcher_name):
    """Resolve a starter safely. The legacy xFIP column currently contains ERA.

    Prefer exact full-name matching. A last-name fallback is allowed only when
    it identifies one unique player, preventing accidental matches for common surnames.
    """
    try:
        if not pitcher_name or pitcher_name == 'Por Anunciar' or df is None or df.empty or 'Name' not in df.columns:
            return None
        names=df['Name'].astype(str)
        exact=df[names.str.casefold()==str(pitcher_name).casefold()]
        match=exact
        if match.empty:
            last=str(pitcher_name).split()[-1].casefold()
            fallback=df[names.str.split().str[-1].str.casefold()==last]
            if fallback['Name'].nunique()!=1:
                return None
            match=fallback
        col='ERA' if 'ERA' in match.columns else ('xFIP' if 'xFIP' in match.columns else None)
        if col is None:
            return None
        val=pd.to_numeric(match.iloc[-1][col],errors='coerce')
        return None if pd.isna(val) else float(val)
    except Exception:
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
                    diagnostico_totales = []
                    
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
                            wrc_loc = _current_offensive_index(df_bat, loc_abbr)
                            wrc_vis = _current_offensive_index(df_bat, vis_abbr)
                            
                            pitcher_loc_nombre = datos_partido["pitcher_local"]
                            xfip_loc = _starter_run_prevention(df_pit_ind, pitcher_loc_nombre)
                            
                            if xfip_loc is None:
                                team_pit_loc = df_pit[df_pit['Team'] == loc_abbr]
                                if team_pit_loc.empty: continue
                                xfip_loc = float(team_pit_loc.iloc[-1]['xFIP'])

                            pitcher_vis_nombre = datos_partido["pitcher_visita"]
                            xfip_vis = _starter_run_prevention(df_pit_ind, pitcher_vis_nombre)
                            
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

                            # ML histórico: misma definición de features que en entrenamiento
                            # (fortaleza agregada de temporada anterior + resultados rolling).
                            bat_col_ml = 'OPS_Index' if 'OPS_Index' in df_bat.columns else 'wRC+'
                            pit_col_ml = 'ERA' if 'ERA' in df_pit.columns else 'xFIP'
                            ml_off_loc = _team_prior_stat(df_bat, loc_abbr, bat_col_ml, wrc_loc)
                            ml_off_vis = _team_prior_stat(df_bat, vis_abbr, bat_col_ml, wrc_vis)
                            ml_pit_loc = _team_prior_stat(df_pit, loc_abbr, pit_col_ml, bullpen_loc_era)
                            ml_pit_vis = _team_prior_stat(df_pit, vis_abbr, pit_col_ml, bullpen_vis_era)
                            res_ml = predictor_ml.predecir_partido(
                                loc_abbr, vis_abbr, ml_off_loc, ml_off_vis, ml_pit_loc, ml_pit_vis, park_factor
                            )

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

                            # --- MOTOR ÚNICO DE SELECCIÓN (producción = backtest) ---
                            mkt_loc_scanner, mkt_vis_scanner = no_vig_two_way(cuota_loc, cuota_vis)
                            mkt_over_scanner, mkt_under_scanner = no_vig_two_way(
                                datos_partido.get("cuota_over"), datos_partido.get("cuota_under")
                            )
                            mkt_sp_loc_scanner, mkt_sp_vis_scanner = no_vig_two_way(
                                datos_partido.get("cuota_spread_loc"), datos_partido.get("cuota_spread_vis")
                            )

                            candidatos = [
                                (moneyline_candidate(f"Gana Local ({datos_partido['local']})", prob_ml_loc, prob_mc_loc, cuota_loc, mkt_loc_scanner), None),
                                (moneyline_candidate(f"Gana Visita ({datos_partido['visita']})", prob_ml_vis, prob_mc_vis, cuota_vis, mkt_vis_scanner), None),
                            ]
                            if datos_partido.get("cuota_over") is not None:
                                candidatos.append((total_candidate(f"Over {linea_casino}", prob_ml_over, prob_mc_over, datos_partido.get("cuota_over"), mkt_over_scanner), linea_casino))
                            if datos_partido.get("cuota_under") is not None:
                                candidatos.append((total_candidate(f"Under {linea_casino}", prob_ml_under, prob_mc_under, datos_partido.get("cuota_under"), mkt_under_scanner), linea_casino))
                            if spread_loc is not None and datos_partido.get("cuota_spread_loc") is not None:
                                candidatos.append((runline_candidate(f"Hándicap {spread_loc:+.1f} ({datos_partido['local']})", prob_ml_spread_loc, prob_mc_spread_loc, datos_partido.get("cuota_spread_loc"), mkt_sp_loc_scanner), spread_loc))
                            if spread_vis is not None and datos_partido.get("cuota_spread_vis") is not None:
                                candidatos.append((runline_candidate(f"Hándicap {spread_vis:+.1f} ({datos_partido['visita']})", prob_ml_spread_vis, prob_mc_spread_vis, datos_partido.get("cuota_spread_vis"), mkt_sp_vis_scanner), spread_vis))

                            for cand, market_line in candidatos:
                                if cand is None:
                                    continue
                                if cand.market == 'Totales':
                                    diagnostico_totales.append({
                                        "Partido": f"{datos_partido['visita']} @ {datos_partido['local']}",
                                        "O/U": cand.selection, "ML": round(cand.prob_ml,1), "MC": round(cand.prob_mc,1),
                                        "Combinada": round(cand.probability,1), "Cuota": cand.odds,
                                        "Edge_pp": cand.edge_pp, "EV_pct": cand.ev_pct,
                                        "Estado": "CANDIDATO" if cand.accepted else "NO BET", "Motivo": cand.reason,
                                    })
                                if not cand.accepted:
                                    continue
                                recomendaciones.append({
                                    "Partido": f"{datos_partido['visita']} @ {datos_partido['local']}",
                                    "Mercado": cand.market,
                                    "Apuesta": cand.selection,
                                    "Prob. ML": f"{round(cand.prob_ml,1)}%",
                                    "Prob. MC": f"{round(cand.prob_mc,1)}%",
                                    "Cuota": cand.odds,
                                    "EV+": f"{round(cand.ev_pct,2)}%",
                                    "Stake Kelly": f"{calcular_criterio_kelly(cand.probability, cand.odds)}%",
                                    "_Score": cand.score,
                                    "_Home": datos_partido['local'], "_Away": datos_partido['visita'],
                                    "_Line": market_line, "_ProbCombined": cand.probability,
                                    "_MarketNoVig": cand.market_no_vig, "_Edge": cand.edge_pp,
                                    "_Disagreement": cand.disagreement_pp,
                                    "_StarterHome": datos_partido.get('pitcher_local'), "_StarterAway": datos_partido.get('pitcher_visita'),
                                    "_Park": park_factor, "_Temp": temp_scan, "_Wind": viento_scan, "_WindDir": dir_scan,
                                })

                        except Exception as e:
                            continue
                    
                    if recomendaciones:
                        df_all = pd.DataFrame(recomendaciones).sort_values("_Score", ascending=False).head(3)
                        ledger_rows = []
                        for _, rr in df_all.iterrows():
                            ledger_rows.append({
                                'game_date': datetime.date.today().isoformat(), 'away': rr['_Away'], 'home': rr['_Home'],
                                'market': rr['Mercado'], 'selection': rr['Apuesta'], 'line': rr['_Line'], 'odds': rr['Cuota'],
                                'prob_ml': rr['Prob. ML'], 'prob_mc': rr['Prob. MC'], 'prob_combined': rr['_ProbCombined'],
                                'market_no_vig': rr['_MarketNoVig'], 'edge_pp': rr['_Edge'], 'ev_pct': str(rr['EV+']).replace('%',''),
                                'disagreement_pp': rr['_Disagreement'], 'score': rr['_Score'],
                                'starter_away': rr['_StarterAway'], 'starter_home': rr['_StarterHome'],
                                'park_factor': rr['_Park'], 'temperature_f': rr['_Temp'], 'wind_mph': rr['_Wind'], 'wind_direction': rr['_WindDir'],
                                'model_version': 'v3', 'result_status': 'pending'
                            })
                        try:
                            append_snapshot(ledger_rows)
                        except Exception as ledger_error:
                            st.caption(f"Ledger local no persistió este snapshot: {ledger_error}")
                        visible = [c for c in df_all.columns if not c.startswith('_')]
                        st.dataframe(df_all[visible], use_container_width=True, hide_index=True)
                    else:
                        st.info("No se encontraron apuestas que superen los filtros de valor del scanner hoy.")

                    if diagnostico_totales:
                        st.markdown("### 🧪 Mejor oportunidad O/U analizada")
                        df_tot_diag=pd.DataFrame(diagnostico_totales)
                        df_tot_diag["_rank"]=df_tot_diag["Edge_pp"].fillna(-999)+df_tot_diag["EV_pct"].fillna(-999)
                        st.dataframe(df_tot_diag.sort_values("_rank",ascending=False).head(3).drop(columns=["_rank"]), use_container_width=True, hide_index=True)
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
                        wrc_loc = _current_offensive_index(df_bat, loc_abbr)
                        wrc_vis = _current_offensive_index(df_bat, vis_abbr)
                    except Exception as e:
                        st.error(f"Error procesando índice ofensivo: {e}")
                        st.stop()
                    
                    pitcher_loc_nombre = datos_partido["pitcher_local"]
                    xfip_loc = _starter_run_prevention(df_pit_ind, pitcher_loc_nombre)
                    
                    if xfip_loc is None:
                        team_pit_loc = df_pit[df_pit['Team'] == loc_abbr]
                        if team_pit_loc.empty:
                            st.error("❌ No hay datos de pitcheo reales para el local.")
                            st.stop()
                        xfip_loc = float(team_pit_loc.iloc[-1]['xFIP']) 

                    pitcher_vis_nombre = datos_partido["pitcher_visita"]
                    xfip_vis = _starter_run_prevention(df_pit_ind, pitcher_vis_nombre)
                    
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
                        prob_spread_loc = carreras_dict.get(f"Spread Local {spread_loc:+.1f}", 50.0)
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
                        prob_spread_vis = carreras_dict.get(f"Spread Visita {spread_vis:+.1f}", 50.0)
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
