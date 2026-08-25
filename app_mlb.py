import streamlit as st
import pandas as pd
import requests
import numpy as np
import os
import datetime
import math

from modules.montecarlo_mlb import simular_partido_mlb
from modules.odds_mlb import analizar_apuestas_mlb
from modules.ml_mlb import PredictorMLMLB, preferred_batting_column, preferred_pitching_column
from modules.blend_calibration import market_blend_weights
from modules.scanner_engine import (
    no_vig_two_way, moneyline_candidate, total_candidate, runline_candidate
)
from modules.pick_ledger import append_snapshot, persistent_backend_available
from modules.game_context import (
    slate_date, park_for_team, match_odds_game, market_from_event, conservative_auto_weather, best_auto_weather
)

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="MLB Quant Analytics", layout="wide", page_icon="⚾")

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
    return ""

ODDS_API_KEY = _get_odds_api_key()

EQUIPOS_MAP = {
    "New York Yankees": "NYY", "Boston Red Sox": "BOS", "Los Angeles Dodgers": "LAD",
    "Houston Astros": "HOU", "Atlanta Braves": "ATL", "Philadelphia Phillies": "PHI",
    "Baltimore Orioles": "BAL", "Tampa Bay Rays": "TB", "Toronto Blue Jays": "TOR",
    "Chicago White Sox": "CWS", "Cleveland Guardians": "CLE", "Detroit Tigers": "DET",
    "Kansas City Royals": "KC", "Minnesota Twins": "MIN", "Los Angeles Angels": "LAA",
    "Oakland Athletics": "OAK", "Athletics": "OAK", "Sacramento Athletics": "OAK",
    "Seattle Mariners": "SEA", "Texas Rangers": "TEX", "Chicago Cubs": "CHC",
    "Cincinnati Reds": "CIN", "Milwaukee Brewers": "MIL", "Pittsburgh Pirates": "PIT",
    "St. Louis Cardinals": "STL", "Arizona Diamondbacks": "AZ", "Colorado Rockies": "COL",
    "San Francisco Giants": "SF", "San Diego Padres": "SD", "Miami Marlins": "MIA",
    "New York Mets": "NYM", "Washington Nationals": "WSH"
}

def american_to_decimal(am_odds):
    try:
        if am_odds is None: return None
        am_odds = float(am_odds)
        if am_odds == 0: return None
        return round((am_odds / 100.0) + 1, 2) if am_odds > 0 else round((100.0 / abs(am_odds)) + 1, 2)
    except Exception:
        return None

def _prob_no_vig_dos_vias(cuota_a, cuota_b):
    try:
        a, b = float(cuota_a), float(cuota_b)
        if a <= 1 or b <= 1: return None, None
        ia, ib = 1.0/a, 1.0/b; total = ia+ib
        return ia/total, ib/total
    except (TypeError, ValueError, ZeroDivisionError): return None, None

def _pasa_valor(prob_pct, cuota, mercado_no_vig=None, min_ev=0.03, min_edge=0.025):
    try:
        p=float(prob_pct)/100.; o=float(cuota); ev=p*o-1.
        if ev<min_ev: return False
        if mercado_no_vig is not None and (p-float(mercado_no_vig))<min_edge: return False
        return True
    except (TypeError, ValueError): return False

def _score_valor(prob_pct, cuota, mercado_no_vig=None, desacuerdo_pp=0.0):
    try:
        p=float(prob_pct)/100.; o=float(cuota); ev_pct=(p*o-1.)*100.; edge_pp=0. if mercado_no_vig is None else (p-float(mercado_no_vig))*100.; return round((1.5*edge_pp)+ev_pct-max(0.,float(desacuerdo_pp))*.15,4)
    except (TypeError, ValueError): return -999.

def _diagnostico_total(prob_ml, prob_mc, prob_comb, cuota, mercado_no_vig, desacuerdo):
    try:
        p=float(prob_comb)/100.; ev=(p*float(cuota)-1.)*100. if cuota is not None else None; edge=(p-float(mercado_no_vig))*100. if mercado_no_vig is not None else None
        checks=[(float(prob_ml)>=52.,f"ML {float(prob_ml):.1f}% < 52%"),(float(prob_mc)>=52.,f"MC {float(prob_mc):.1f}% < 52%"),(float(prob_comb)>=54.,f"Combinada {float(prob_comb):.1f}% < 54%"),(float(desacuerdo)<=15.,f"Desacuerdo {float(desacuerdo):.1f} pp > 15"),(edge is not None and edge>=4.,"Edge < 4 pp"),(ev is not None and ev>=4.,"EV < 4%")]
        fails=[msg for ok,msg in checks if not ok]; return {"EV_pct":None if ev is None else round(ev,2),"Edge_pp":None if edge is None else round(edge,2),"Estado":"CANDIDATO" if not fails else "NO BET","Motivo":"Cumple filtros O/U" if not fails else '; '.join(fails)}
    except Exception as e: return {"EV_pct":None,"Edge_pp":None,"Estado":"NO BET","Motivo":f"Error diagnóstico: {e}"}

def _team_prior_stat(df, team, column, fallback):
    try:
        if df is None or df.empty or column not in df.columns or 'Team' not in df.columns: return float(fallback)
        x=df[df['Team']==team].copy()
        if x.empty: return float(fallback)
        if 'Season' in x.columns:
            x['_SeasonNum']=pd.to_numeric(x['Season'],errors='coerce'); target=slate_date().year-1; eligible=x[x['_SeasonNum']<=target].sort_values('_SeasonNum')
            if not eligible.empty:
                val=pd.to_numeric(eligible.iloc[-1][column],errors='coerce')
                if pd.notna(val): return float(val)
        val=pd.to_numeric(x.iloc[-1][column],errors='coerce'); return float(val) if pd.notna(val) else float(fallback)
    except Exception: return float(fallback)

def _current_offensive_index(df, team, fallback=100.0):
    """Current offense centered at league average, preferring official/FanGraphs indices."""
    try:
        if df is None or df.empty or 'Team' not in df.columns: return float(fallback)
        col=preferred_batting_column(df)
        if col is None: return float(fallback)
        x=df.copy(); x['_v']=pd.to_numeric(x[col],errors='coerce')
        if 'Season' in x.columns:
            x['_s']=pd.to_numeric(x['Season'],errors='coerce'); latest=x['_s'].dropna().max(); season=x[x['_s']==latest]
        else: season=x
        team_rows=season[season['Team']==team]
        if team_rows.empty: team_rows=x[x['Team']==team]
        vals=team_rows['_v'].dropna()
        if vals.empty: return float(fallback)
        team_val=float(vals.iloc[-1])
        if col in ('wRC+','Offense_Index'):
            return float(np.clip(team_val,70.,130.))
        league=season['_v'].dropna(); center=float(league.median()) if len(league) else float(x['_v'].dropna().median())
        if not center or pd.isna(center): return float(fallback)
        return float(np.clip((team_val/center)*100.,75.,125.))
    except Exception: return float(fallback)

def _starter_run_prevention(df, pitcher_name):
    try:
        if not pitcher_name or pitcher_name=='Por Anunciar' or df is None or df.empty or 'Name' not in df.columns: return None
        names=df['Name'].astype(str); exact=df[names.str.casefold()==str(pitcher_name).casefold()]; match=exact
        if match.empty:
            last=str(pitcher_name).split()[-1].casefold(); fallback=df[names.str.split().str[-1].str.casefold()==last]
            if fallback['Name'].nunique()!=1: return None
            match=fallback
        col=next((c for c in ('xFIP','FIP','ERA') if c in match.columns and pd.to_numeric(match[c],errors='coerce').notna().any()),None)
        if col is None: return None
        val=pd.to_numeric(match.iloc[-1][col],errors='coerce'); return None if pd.isna(val) else float(val)
    except Exception: return None

def _starter_expected_innings(df, pitcher_name, default=5.2):
    try:
        if not pitcher_name or pitcher_name=='Por Anunciar' or df is None or df.empty or 'Name' not in df.columns: return float(default)
        names=df['Name'].astype(str); match=df[names.str.casefold()==str(pitcher_name).casefold()]
        if match.empty:
            last=str(pitcher_name).split()[-1].casefold(); alt=df[names.str.split().str[-1].str.casefold()==last]
            if alt['Name'].nunique()!=1: return float(default)
            match=alt
        ip=pd.to_numeric(match.get('IP'),errors='coerce') if 'IP' in match.columns else pd.Series(dtype=float); gs=pd.to_numeric(match.get('GS'),errors='coerce') if 'GS' in match.columns else pd.Series(dtype=float)
        if len(ip) and len(gs) and pd.notna(ip.iloc[-1]) and pd.notna(gs.iloc[-1]) and float(gs.iloc[-1])>0: return float(np.clip(float(ip.iloc[-1])/float(gs.iloc[-1]),3.5,6.8))
        return float(default)
    except Exception: return float(default)

@st.cache_data(ttl=3600)
def _load_bullpen_proxy():
    try:
        path='data/mlb_bullpen.csv'
        if not os.path.exists(path): return pd.DataFrame()
        df=pd.read_csv(path)
        if df.empty or 'Team' not in df.columns or 'ERA' not in df.columns: return pd.DataFrame()
        df['ERA']=pd.to_numeric(df['ERA'],errors='coerce')
        if 'Season' in df.columns: df['Season']=pd.to_numeric(df['Season'],errors='coerce')
        return df.dropna(subset=['Team','ERA'])
    except Exception: return pd.DataFrame()

def _bullpen_era(team, fallback):
    try:
        df=_load_bullpen_proxy(); rows=df[df['Team'].astype(str).str.upper()==str(team).upper()]
        if rows.empty: return float(fallback)
        if 'Season' in rows.columns: rows=rows.sort_values('Season')
        return float(rows.iloc[-1]['ERA'])
    except Exception: return float(fallback)

def calcular_criterio_kelly(probabilidad_real, cuota_decimal, fraccion=0.25, prob_push=0.0):
    try:
        if cuota_decimal is None or probabilidad_real is None: return 0.0
        p=max(0.,float(probabilidad_real)/100.); push=max(0.,float(prob_push)/100.); q=max(0.,1.-p-push); b=float(cuota_decimal)-1.; decisions=p+q
        if b<=0 or decisions<=0:return 0.0
        return round(max(0.,((b*p-q)/(b*decisions))*fraccion)*100.,2)
    except Exception:return 0.0

def estimar_prob_ml(proyeccion, linea, tipo="over", sigma=None):
    if proyeccion is None or linea is None:return 50.0
    try:
        sigma=float(sigma) if sigma is not None else (3.5 if tipo in ['over','under'] else 4.2); sigma=max(1.,sigma)
        if tipo=='over':z=(proyeccion-linea)/sigma
        elif tipo=='under':z=(linea-proyeccion)/sigma
        elif tipo in ['spread_loc','spread_vis']:z=(proyeccion+linea)/sigma
        else:return 50.0
        prob=.5*(1.+math.erf(z/math.sqrt(2.)));return max(0.,min(100.,round(prob*100.,2)))
    except Exception:return 50.0

@st.cache_data(ttl=3600)
def cargar_datos_historicos():
    bateo=pd.DataFrame();pitcheo=pd.DataFrame();pitcheo_individual=pd.DataFrame();park=pd.DataFrame();games=pd.DataFrame()
    try:
        if os.path.exists('data/mlb_batting.csv'):
            df_temp=pd.read_csv('data/mlb_batting.csv',sep=None,engine='python',on_bad_lines='skip');df_temp.columns=df_temp.columns.str.strip()
            if not df_temp.empty and 'Team' in df_temp.columns:
                metric=preferred_batting_column(df_temp)
                if metric: df_temp[metric]=pd.to_numeric(df_temp[metric],errors='coerce');bateo=df_temp.dropna(subset=[metric])
        if os.path.exists('data/mlb_pitching.csv'):
            df_temp=pd.read_csv('data/mlb_pitching.csv',sep=None,engine='python',on_bad_lines='skip');df_temp.columns=df_temp.columns.str.strip()
            if not df_temp.empty and 'Team' in df_temp.columns:
                for c in ('xFIP','FIP','ERA'):
                    if c in df_temp.columns:df_temp[c]=pd.to_numeric(df_temp[c],errors='coerce')
                metric=preferred_pitching_column(df_temp)
                if metric:pitcheo=df_temp.dropna(subset=[metric])
        if os.path.exists('data/mlb_pitching_individual.csv'):
            df_temp=pd.read_csv('data/mlb_pitching_individual.csv',sep=None,engine='python',on_bad_lines='skip');df_temp.columns=df_temp.columns.str.strip()
            if not df_temp.empty and 'Name' in df_temp.columns:
                for c in ('xFIP','FIP','ERA','IP','GS'):
                    if c in df_temp.columns:df_temp[c]=pd.to_numeric(df_temp[c],errors='coerce')
                pitcheo_individual=df_temp
        if os.path.exists('data/mlb_park_factors.csv'):
            df_temp=pd.read_csv('data/mlb_park_factors.csv',sep=None,engine='python',on_bad_lines='skip');df_temp.columns=df_temp.columns.str.strip();park=df_temp if not df_temp.empty else park
        if os.path.exists('data/mlb_games.csv'):
            df_temp=pd.read_csv('data/mlb_games.csv',sep=None,engine='python',on_bad_lines='skip');df_temp.columns=df_temp.columns.str.strip();games=df_temp if not df_temp.empty else games
    except Exception as e:st.warning(f'Aviso menor de lectura de archivos: {e}')
    return bateo,pitcheo,pitcheo_individual,park,games

def _make_predictor(bat,pit,games):
    p=PredictorMLMLB();p.entrenar(bat,pit,games);return p

@st.cache_resource(show_spinner=False)
def _cached_predictor(sig_bat,sig_pit,sig_games,_bat,_pit,_games):return _make_predictor(_bat,_pit,_games)
def _df_signature(df):
    if df is None or df.empty:return (0,())
    try:return (len(df),tuple(df.columns),str(pd.util.hash_pandas_object(df,index=True).sum()))
    except Exception:return (len(df),tuple(df.columns))

df_bat,df_pit,df_pit_ind,df_parks,df_games=cargar_datos_historicos()
predictor_ml=_cached_predictor(_df_signature(df_bat),_df_signature(df_pit),_df_signature(df_games),df_bat,df_pit,df_games)

def obtener_clima_estadio(nombre_equipo):
    ciudades={"New York Yankees":"New_York","Boston Red Sox":"Boston","Los Angeles Dodgers":"Los_Angeles","Houston Astros":"Houston","Atlanta Braves":"Atlanta","Philadelphia Phillies":"Philadelphia","Baltimore Orioles":"Baltimore","Tampa Bay Rays":"St_Petersburg","Toronto Blue Jays":"Toronto","Chicago White Sox":"Chicago","Cleveland Guardians":"Cleveland","Detroit Tigers":"Detroit","Kansas City Royals":"Kansas_City","Minnesota Twins":"Minneapolis","Los Angeles Angels":"Anaheim","Oakland Athletics":"Sacramento","Athletics":"Sacramento","Sacramento Athletics":"Sacramento","Seattle Mariners":"Seattle","Texas Rangers":"Arlington","Chicago Cubs":"Chicago","Cincinnati Reds":"Cincinnati","Milwaukee Brewers":"Milwaukee","Pittsburgh Pirates":"Pittsburgh","St. Louis Cardinals":"St_Louis","Arizona Diamondbacks":"Phoenix","Colorado Rockies":"Denver","San Francisco Giants":"San_Francisco","San Diego Padres":"San_Diego","Miami Marlins":"Miami","New York Mets":"New_York","Washington Nationals":"Washington"}
    ciudad=ciudades.get(nombre_equipo)
    if not ciudad:return None,None,'None'
    try:
        res=requests.get(f'https://wttr.in/{ciudad}?format=j1',timeout=5)
        if res.status_code==200:
            curr=res.json().get('current_condition',[{}])[0];temp_f=int(curr.get('temp_F'));wind_mph=int(curr.get('windspeedMiles'));wind_dir=curr.get('winddir16Point','');return temp_f,wind_mph,f'Compass {wind_dir}' if wind_dir else 'None'
    except Exception:pass
    return None,None,'None'

@st.cache_data(ttl=300)
def obtener_cartelera_y_cuotas_automaticas():
    hoy=slate_date().strftime('%Y-%m-%d');url_mlb=f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={hoy}&hydrate=probablePitcher,team';partidos={}
    try:
        res_mlb=requests.get(url_mlb,timeout=5)
        if res_mlb.status_code==200:
            for date_item in res_mlb.json().get('dates',[]):
                for game in date_item.get('games',[]):
                    home=game.get('teams',{}).get('home',{}).get('team',{}).get('name','');away=game.get('teams',{}).get('away',{}).get('team',{}).get('name','');hp=game.get('teams',{}).get('home',{}).get('probablePitcher',{}).get('fullName','Por Anunciar');ap=game.get('teams',{}).get('away',{}).get('probablePitcher',{}).get('fullName','Por Anunciar')
                    if home and away:
                        game_pk=game.get('gamePk');key=f'⚾ {away} ({ap}) @ {home} ({hp}) · #{game_pk}';partidos[key]={'local':home,'visita':away,'pitcher_local':hp,'pitcher_visita':ap,'game_pk':game_pk,'start_time_utc':game.get('gameDate'),'linea_carreras':None,'cuota_loc':None,'cuota_vis':None,'cuota_over':None,'cuota_under':None,'spread_loc':None,'cuota_spread_loc':None,'spread_vis':None,'cuota_spread_vis':None}
    except Exception as e:st.error(f'Error en MLB StatsAPI: {e}')
    if ODDS_API_KEY:
        try:
            res_odds=requests.get(f'https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/?apiKey={ODDS_API_KEY}&regions=us&markets=h2h,totals,spreads&oddsFormat=american',timeout=5)
            if res_odds.status_code==200:
                data_odds=res_odds.json()
                for p_game in partidos.values():
                    event=match_odds_game(data_odds,p_game)
                    if event:p_game.update(market_from_event(event,american_to_decimal))
            elif res_odds.status_code==429:st.warning('⚠️ Límite de tu API Key de cuotas agotado.')
            elif res_odds.status_code==401:st.warning('⚠️ API Key de The-Odds-API rechazada.')
            else:st.warning(f'⚠️ Error de servidor de cuotas: {res_odds.status_code}')
        except Exception as e:st.warning(f'Aviso de sincronización de cuotas: {e}')
    return partidos

st.title('⚾ MLB Quant Analytics Pro V6')
st.markdown('Sistema autónomo de Sabermetría, Clima, Simulación Monte Carlo, Escáner Global y Criterio de Kelly.')

if df_bat.empty or df_pit.empty:
    st.warning('⚠️ Faltan datos históricos. Ejecuta `minero_mlb.py`.')
else:
    partidos_hoy=obtener_cartelera_y_cuotas_automaticas()
    if not partidos_hoy:st.info('No hay partidos programados para hoy.')
    else:
        if not ODDS_API_KEY:st.warning('⚠️ ODDS_API_KEY no configurada en Secrets/entorno. La cartelera se muestra, pero no se emitirán apuestas sin cuotas reales.')
        if not persistent_backend_available():st.caption('ℹ️ Ledger en modo local: configura GITHUB_TOKEN y LEDGER_GITHUB_REPO para persistencia entre reinicios.')
        modo_app=st.sidebar.radio('Modo de Operación',['🎯 Análisis Individual por Partido','🔍 Escáner Automático de la Jornada (EV+)'])
        if modo_app=='🔍 Escáner Automático de la Jornada (EV+)':
            st.subheader('🔍 Escáner Cuántico de Valor para Toda la Jornada');st.markdown('Escanea la cartelera con filtros calibrados por mercado: consenso ML + Monte Carlo, no-vig, edge, EV y desacuerdo máximo.')
            if st.button('🚀 Ejecutar Escáner Global de la Jornada',type='primary'):
                with st.spinner('Escaneando duelos y procesando simulaciones avanzadas...'):
                    recomendaciones=[];diagnostico_totales=[];blend_weights=market_blend_weights();errores_datos=[]
                    for llave,datos_partido in partidos_hoy.items():
                        loc_abbr=EQUIPOS_MAP.get(datos_partido['local'],'');vis_abbr=EQUIPOS_MAP.get(datos_partido['visita'],'')
                        if not loc_abbr or not vis_abbr:errores_datos.append({'Partido':f"{datos_partido.get('visita','?')} @ {datos_partido.get('local','?')}",'Error':'Equipo no normalizable'});continue
                        cuota_loc=datos_partido.get('cuota_loc');cuota_vis=datos_partido.get('cuota_vis');linea_casino=datos_partido.get('linea_carreras')
                        if cuota_loc is None or cuota_vis is None or linea_casino is None:errores_datos.append({'Partido':f"{datos_partido['visita']} @ {datos_partido['local']}",'Error':'Cuotas/total no disponibles o no emparejados de forma segura'});continue
                        try:
                            wrc_loc=_current_offensive_index(df_bat,loc_abbr);wrc_vis=_current_offensive_index(df_bat,vis_abbr);pln=datos_partido['pitcher_local'];xfip_loc=_starter_run_prevention(df_pit_ind,pln);starter_ip_loc=_starter_expected_innings(df_pit_ind,pln)
                            if xfip_loc is None:errores_datos.append({'Partido':f"{datos_partido['visita']} @ {datos_partido['local']}",'Error':f'Abridor local sin métrica individual fiable: {pln}'});continue
                            pvn=datos_partido['pitcher_visita'];xfip_vis=_starter_run_prevention(df_pit_ind,pvn);starter_ip_vis=_starter_expected_innings(df_pit_ind,pvn)
                            if xfip_vis is None:errores_datos.append({'Partido':f"{datos_partido['visita']} @ {datos_partido['local']}",'Error':f'Abridor visitante sin métrica individual fiable: {pvn}'});continue
                            th=df_pit[df_pit['Team']==loc_abbr];tv=df_pit[df_pit['Team']==vis_abbr];bp_h=_bullpen_era(loc_abbr,float(th.iloc[-1]['ERA']) if not th.empty else 4.0);bp_v=_bullpen_era(vis_abbr,float(tv.iloc[-1]['ERA']) if not tv.empty else 4.0);park_info=park_for_team(df_parks,loc_abbr)
                            if not park_info:errores_datos.append({'Partido':f"{datos_partido['visita']} @ {datos_partido['local']}",'Error':'Parque no resoluble'});continue
                            park_factor=park_info['park_factor'];altitud=park_info['altitude_ft'];tr,wr,dr=obtener_clima_estadio(datos_partido['local']);temp_scan,viento_scan,dir_scan,weather_source=best_auto_weather(datos_partido['local'],datos_partido.get('start_time_utc'),tr,wr,dr)
                            res_mc=simular_partido_mlb(local=datos_partido['local'],visita=datos_partido['visita'],pitcher_loc_xfip=xfip_loc,pitcher_vis_xfip=xfip_vis,wrc_loc=wrc_loc,wrc_vis=wrc_vis,bullpen_loc_era=bp_h,bullpen_vis_era=bp_v,park_factor=park_factor,altitud_ft=altitud,viento_mph=viento_scan,direccion_viento=dir_scan,temp_f=temp_scan,linea_carreras_casino=linea_casino,df_games=df_games,num_simulaciones=50000,starter_ip_loc=starter_ip_loc,starter_ip_vis=starter_ip_vis)
                            bat_col_ml=preferred_batting_column(df_bat) or 'OPS_Index';pit_col_ml=preferred_pitching_column(df_pit) or 'ERA';ml_off_loc=_team_prior_stat(df_bat,loc_abbr,bat_col_ml,wrc_loc);ml_off_vis=_team_prior_stat(df_bat,vis_abbr,bat_col_ml,wrc_vis);ml_pit_loc=_team_prior_stat(df_pit,loc_abbr,pit_col_ml,bp_h);ml_pit_vis=_team_prior_stat(df_pit,vis_abbr,pit_col_ml,bp_v);res_ml=predictor_ml.predecir_partido(loc_abbr,vis_abbr,ml_off_loc,ml_off_vis,ml_pit_loc,ml_pit_vis,park_factor)
                            prob_mc_loc=res_mc['Moneyline']['Gana Local'];prob_mc_vis=res_mc['Moneyline']['Gana Visita'];carreras_dict=res_mc.get('Carreras',{});prob_mc_over=carreras_dict.get(f'Over {linea_casino}',50.);prob_mc_under=carreras_dict.get(f'Under {linea_casino}',50.);spread_loc=datos_partido.get('spread_loc');spread_vis=datos_partido.get('spread_vis');prob_mc_spread_loc=carreras_dict.get(f'Spread Local {spread_loc:+.1f}',50.) if spread_loc is not None else 50.;prob_mc_spread_vis=carreras_dict.get(f'Spread Visita {spread_vis:+.1f}',50.) if spread_vis is not None else 50.;prob_ml_loc=res_ml['Probabilidad_Local'];prob_ml_vis=res_ml['Probabilidad_Visita'];proy=res_ml.get('Proyeccion_Carreras',linea_casino);prob_ml_over=estimar_prob_ml(proy,linea_casino,'over',res_ml.get('Sigma_Carreras'));prob_ml_under=estimar_prob_ml(proy,linea_casino,'under',res_ml.get('Sigma_Carreras'));ph=res_ml.get('Proyeccion_Handicap_Local',0);prob_ml_spread_loc=estimar_prob_ml(ph,spread_loc,'spread_loc',res_ml.get('Sigma_Handicap')) if spread_loc is not None else 50.;prob_ml_spread_vis=estimar_prob_ml(-ph,spread_vis,'spread_vis',res_ml.get('Sigma_Handicap')) if spread_vis is not None else 50.;mhl,mhv=no_vig_two_way(cuota_loc,cuota_vis);mo,mu=no_vig_two_way(datos_partido.get('cuota_over'),datos_partido.get('cuota_under'));msl,msv=no_vig_two_way(datos_partido.get('cuota_spread_loc'),datos_partido.get('cuota_spread_vis'))
                            candidatos=[(moneyline_candidate(f"Gana Local ({datos_partido['local']})",prob_ml_loc,prob_mc_loc,cuota_loc,mhl,blend_weight_ml=blend_weights['Moneyline']['ml_weight']),None),(moneyline_candidate(f"Gana Visita ({datos_partido['visita']})",prob_ml_vis,prob_mc_vis,cuota_vis,mhv,blend_weight_ml=blend_weights['Moneyline']['ml_weight']),None)]
                            if datos_partido.get('cuota_over') is not None:candidatos.append((total_candidate(f'Over {linea_casino}',prob_ml_over,prob_mc_over,datos_partido.get('cuota_over'),mo,carreras_dict.get(f'Push {linea_casino}',0.),blend_weight_ml=blend_weights['Totales']['ml_weight']),linea_casino))
                            if datos_partido.get('cuota_under') is not None:candidatos.append((total_candidate(f'Under {linea_casino}',prob_ml_under,prob_mc_under,datos_partido.get('cuota_under'),mu,carreras_dict.get(f'Push {linea_casino}',0.),blend_weight_ml=blend_weights['Totales']['ml_weight']),linea_casino))
                            if spread_loc is not None and datos_partido.get('cuota_spread_loc') is not None:candidatos.append((runline_candidate(f"Hándicap {spread_loc:+.1f} ({datos_partido['local']})",prob_ml_spread_loc,prob_mc_spread_loc,datos_partido.get('cuota_spread_loc'),msl,carreras_dict.get(f'Push Spread Local {spread_loc:+.1f}',0.),blend_weight_ml=blend_weights['Hándicap']['ml_weight']),spread_loc))
                            if spread_vis is not None and datos_partido.get('cuota_spread_vis') is not None:candidatos.append((runline_candidate(f"Hándicap {spread_vis:+.1f} ({datos_partido['visita']})",prob_ml_spread_vis,prob_mc_spread_vis,datos_partido.get('cuota_spread_vis'),msv,carreras_dict.get(f'Push Spread Visita {spread_vis:+.1f}',0.),blend_weight_ml=blend_weights['Hándicap']['ml_weight']),spread_vis))
                            for cand,market_line in candidatos:
                                if cand is None:continue
                                if cand.market=='Totales':diagnostico_totales.append({'Partido':f"{datos_partido['visita']} @ {datos_partido['local']}",'O/U':cand.selection,'ML':round(cand.prob_ml,1),'MC':round(cand.prob_mc,1),'Combinada':round(cand.probability,1),'Cuota':cand.odds,'Edge_pp':cand.edge_pp,'EV_pct':cand.ev_pct,'Estado':'CANDIDATO' if cand.accepted else 'NO BET','Motivo':cand.reason})
                                if not cand.accepted:continue
                                recomendaciones.append({'Partido':f"{datos_partido['visita']} @ {datos_partido['local']}",'Mercado':cand.market,'Apuesta':cand.selection,'Prob. ML':f'{round(cand.prob_ml,1)}%','Prob. MC':f'{round(cand.prob_mc,1)}%','Cuota':cand.odds,'EV+':f'{round(cand.ev_pct,2)}%','Stake Kelly':f'{calcular_criterio_kelly(cand.probability,cand.odds,prob_push=cand.push_probability)}%','_Score':cand.score,'_Home':datos_partido['local'],'_Away':datos_partido['visita'],'_GamePk':datos_partido.get('game_pk'),'_Line':market_line,'_ProbCombined':cand.probability,'_MarketNoVig':cand.market_no_vig,'_Edge':cand.edge_pp,'_Disagreement':cand.disagreement_pp,'_BlendWeightML':cand.blend_weight_ml,'_BatMetric':res_ml.get('Metrica_Bateo'),'_PitMetric':res_ml.get('Metrica_Pitcheo'),'_StarterHome':datos_partido.get('pitcher_local'),'_StarterAway':datos_partido.get('pitcher_visita'),'_StarterIPHome':starter_ip_loc,'_StarterIPAway':starter_ip_vis,'_Park':park_factor,'_Temp':temp_scan,'_Wind':viento_scan,'_WindDir':dir_scan,'_WeatherSource':weather_source})
                        except Exception as e:errores_datos.append({'Partido':f"{datos_partido.get('visita','?')} @ {datos_partido.get('local','?')}",'Error':str(e)[:180]});continue
                    if recomendaciones:
                        df_all=pd.DataFrame(recomendaciones).sort_values('_Score',ascending=False).head(3);ledger_rows=[]
                        for _,rr in df_all.iterrows():ledger_rows.append({'game_date':slate_date().isoformat(),'game_pk':rr['_GamePk'],'away':rr['_Away'],'home':rr['_Home'],'market':rr['Mercado'],'selection':rr['Apuesta'],'line':rr['_Line'],'odds':rr['Cuota'],'prob_ml':rr['Prob. ML'],'prob_mc':rr['Prob. MC'],'prob_combined':rr['_ProbCombined'],'blend_weight_ml':rr['_BlendWeightML'],'market_no_vig':rr['_MarketNoVig'],'edge_pp':rr['_Edge'],'ev_pct':str(rr['EV+']).replace('%',''),'disagreement_pp':rr['_Disagreement'],'score':rr['_Score'],'batting_metric':rr['_BatMetric'],'pitching_metric':rr['_PitMetric'],'starter_away':rr['_StarterAway'],'starter_home':rr['_StarterHome'],'starter_ip_away':rr['_StarterIPAway'],'starter_ip_home':rr['_StarterIPHome'],'park_factor':rr['_Park'],'temperature_f':rr['_Temp'],'wind_mph':rr['_Wind'],'wind_direction':rr['_WindDir'],'weather_source':rr['_WeatherSource'],'model_version':'v6','result_status':'pending'})
                        try:append_snapshot(ledger_rows)
                        except Exception as le:st.caption(f'Ledger local no persistió este snapshot: {le}')
                        st.dataframe(df_all[[c for c in df_all.columns if not c.startswith('_')]],use_container_width=True,hide_index=True)
                    else:st.info('No se encontraron apuestas que superen los filtros de valor del scanner hoy.')
                    if diagnostico_totales:
                        st.markdown('### 🧪 Mejor oportunidad O/U analizada');d=pd.DataFrame(diagnostico_totales);d['_rank']=d['Edge_pp'].fillna(-999)+d['EV_pct'].fillna(-999);st.dataframe(d.sort_values('_rank',ascending=False).head(3).drop(columns=['_rank']),use_container_width=True,hide_index=True)
                    if errores_datos:
                        with st.expander(f'⚠️ Partidos no evaluados por datos incompletos ({len(errores_datos)})'):st.dataframe(pd.DataFrame(errores_datos),use_container_width=True,hide_index=True)
        else:
            st.subheader('1. Cartelera Oficial Sincronizada');seleccion=st.selectbox('Selecciona un duelo:',list(partidos_hoy.keys()));datos_partido=partidos_hoy[seleccion];ct,cw,cd=obtener_clima_estadio(datos_partido['local']);temp_auto,viento_auto,dir_auto,weather_auto_source=best_auto_weather(datos_partido['local'],datos_partido.get('start_time_utc'),ct,cw,cd);st.subheader('2. Datos del Mercado y Clima (En Vivo)');c1,c2,c3,c4=st.columns(4);opciones_viento=['None','Outfield (Hacia Afuera)','Infield (Hacia Adentro)','Lateral (Derecha a Izquierda)','Lateral (Izquierda a Derecha)'];indice_dir=opciones_viento.index(dir_auto) if dir_auto in opciones_viento else 0;st.caption(f'Fuente clima automática: {weather_auto_source}')
            with c1:st.metric('Línea O/U Casino',datos_partido['linea_carreras'] if datos_partido['linea_carreras'] is not None else 'No disponible');st.metric('Cuota Over',datos_partido['cuota_over'] if datos_partido['cuota_over'] is not None else 'No disponible')
            with c2:st.metric(f"Cuota ML ({datos_partido['local']})",datos_partido['cuota_loc'] if datos_partido['cuota_loc'] is not None else 'No disponible');st.metric(f"Cuota ML ({datos_partido['visita']})",datos_partido['cuota_vis'] if datos_partido['cuota_vis'] is not None else 'No disponible')
            with c3:viento=st.number_input('Viento (mph)',value=int(viento_auto) if viento_auto is not None else 8,step=1);dir_viento=st.selectbox('Dirección del Viento',opciones_viento,index=indice_dir)
            with c4:temp=st.slider('Temperatura (°F)',30,110,int(temp_auto) if temp_auto is not None else 72)
            if st.button('🚀 Ejecutar Simulación Cuántica',type='primary'):
                with st.spinner('Procesando datos en vivo y ejecutando simulaciones...'):
                    loc_abbr=EQUIPOS_MAP.get(datos_partido['local'],'');vis_abbr=EQUIPOS_MAP.get(datos_partido['visita'],'')
                    if df_bat.empty or df_pit.empty or df_parks.empty:st.error('❌ Error crítico: Las bases de datos históricas están vacías.');st.stop()
                    wrc_loc=_current_offensive_index(df_bat,loc_abbr);wrc_vis=_current_offensive_index(df_bat,vis_abbr);pln=datos_partido['pitcher_local'];xfip_loc=_starter_run_prevention(df_pit_ind,pln);starter_ip_loc=_starter_expected_innings(df_pit_ind,pln)
                    if xfip_loc is None:st.error(f'❌ Abridor local sin métrica individual fiable: {pln}. No se emite apuesta con un proxy de equipo.');st.stop()
                    pvn=datos_partido['pitcher_visita'];xfip_vis=_starter_run_prevention(df_pit_ind,pvn);starter_ip_vis=_starter_expected_innings(df_pit_ind,pvn)
                    if xfip_vis is None:st.error(f'❌ Abridor visitante sin métrica individual fiable: {pvn}. No se emite apuesta con un proxy de equipo.');st.stop()
                    th=df_pit[df_pit['Team']==loc_abbr];tv=df_pit[df_pit['Team']==vis_abbr];bp_h=_bullpen_era(loc_abbr,float(th.iloc[-1]['ERA']) if not th.empty else 4.0);bp_v=_bullpen_era(vis_abbr,float(tv.iloc[-1]['ERA']) if not tv.empty else 4.0);park_info=park_for_team(df_parks,loc_abbr)
                    if not park_info:st.error(f"❌ Error de integridad: no se pudo resolver el parque de {datos_partido['local']} ({loc_abbr}).");st.stop()
                    park_factor=park_info['park_factor'];altitud=park_info['altitude_ft'];linea_casino=datos_partido.get('linea_carreras')
                    if linea_casino is None:st.error('❌ No hay línea O/U real disponible para este juego.');st.stop()
                    res_mc=simular_partido_mlb(local=datos_partido['local'],visita=datos_partido['visita'],pitcher_loc_xfip=xfip_loc,pitcher_vis_xfip=xfip_vis,wrc_loc=wrc_loc,wrc_vis=wrc_vis,bullpen_loc_era=bp_h,bullpen_vis_era=bp_v,park_factor=park_factor,altitud_ft=altitud,viento_mph=viento,direccion_viento=dir_viento,temp_f=temp,linea_carreras_casino=linea_casino,df_games=df_games,num_simulaciones=50000,starter_ip_loc=starter_ip_loc,starter_ip_vis=starter_ip_vis);bat_col_ml=preferred_batting_column(df_bat) or 'OPS_Index';pit_col_ml=preferred_pitching_column(df_pit) or 'ERA';res_ml=predictor_ml.predecir_partido(loc_abbr,vis_abbr,_team_prior_stat(df_bat,loc_abbr,bat_col_ml,wrc_loc),_team_prior_stat(df_bat,vis_abbr,bat_col_ml,wrc_vis),_team_prior_stat(df_pit,loc_abbr,pit_col_ml,bp_h),_team_prior_stat(df_pit,vis_abbr,pit_col_ml,bp_v),park_factor);carreras_dict=res_mc.get('Carreras',{});prob_mc_loc=res_mc['Moneyline']['Gana Local'];prob_mc_vis=res_mc['Moneyline']['Gana Visita'];prob_ml_loc=res_ml['Probabilidad_Local'];prob_ml_vis=res_ml['Probabilidad_Visita'];prob_mc_over=carreras_dict.get(f'Over {linea_casino}',50.);prob_mc_under=carreras_dict.get(f'Under {linea_casino}',50.);proy=res_ml.get('Proyeccion_Carreras',linea_casino);prob_ml_over=estimar_prob_ml(proy,linea_casino,'over',res_ml.get('Sigma_Carreras'));prob_ml_under=estimar_prob_ml(proy,linea_casino,'under',res_ml.get('Sigma_Carreras'));spread_loc=datos_partido.get('spread_loc');spread_vis=datos_partido.get('spread_vis');prob_mc_spread_loc=carreras_dict.get(f'Spread Local {spread_loc:+.1f}',50.) if spread_loc is not None else 50.;prob_mc_spread_vis=carreras_dict.get(f'Spread Visita {spread_vis:+.1f}',50.) if spread_vis is not None else 50.;ph=res_ml.get('Proyeccion_Handicap_Local',0);prob_ml_spread_loc=estimar_prob_ml(ph,spread_loc,'spread_loc',res_ml.get('Sigma_Handicap')) if spread_loc is not None else 50.;prob_ml_spread_vis=estimar_prob_ml(-ph,spread_vis,'spread_vis',res_ml.get('Sigma_Handicap')) if spread_vis is not None else 50.;cl=datos_partido.get('cuota_loc');cv=datos_partido.get('cuota_vis');co=datos_partido.get('cuota_over');cu=datos_partido.get('cuota_under')
                    if None in (cl,cv,co,cu):st.error('❌ Faltan cuotas reales de ambos lados.');st.stop()
                    mhl,mhv=no_vig_two_way(cl,cv);mo,mu=no_vig_two_way(co,cu);msl,msv=no_vig_two_way(datos_partido.get('cuota_spread_loc'),datos_partido.get('cuota_spread_vis'));blend_weights=market_blend_weights();candidatos_ind=[moneyline_candidate(f"Gana Local ({datos_partido['local']})",prob_ml_loc,prob_mc_loc,cl,mhl,blend_weight_ml=blend_weights['Moneyline']['ml_weight']),moneyline_candidate(f"Gana Visita ({datos_partido['visita']})",prob_ml_vis,prob_mc_vis,cv,mhv,blend_weight_ml=blend_weights['Moneyline']['ml_weight']),total_candidate(f'Over {linea_casino}',prob_ml_over,prob_mc_over,co,mo,carreras_dict.get(f'Push {linea_casino}',0.),blend_weight_ml=blend_weights['Totales']['ml_weight']),total_candidate(f'Under {linea_casino}',prob_ml_under,prob_mc_under,cu,mu,carreras_dict.get(f'Push {linea_casino}',0.),blend_weight_ml=blend_weights['Totales']['ml_weight'])]
                    if spread_loc is not None and datos_partido.get('cuota_spread_loc') is not None:candidatos_ind.append(runline_candidate(f"Hándicap {spread_loc:+.1f} ({datos_partido['local']})",prob_ml_spread_loc,prob_mc_spread_loc,datos_partido['cuota_spread_loc'],msl,carreras_dict.get(f'Push Spread Local {spread_loc:+.1f}',0.),blend_weight_ml=blend_weights['Hándicap']['ml_weight']))
                    if spread_vis is not None and datos_partido.get('cuota_spread_vis') is not None:candidatos_ind.append(runline_candidate(f"Hándicap {spread_vis:+.1f} ({datos_partido['visita']})",prob_ml_spread_vis,prob_mc_spread_vis,datos_partido['cuota_spread_vis'],msv,carreras_dict.get(f'Push Spread Visita {spread_vis:+.1f}',0.),blend_weight_ml=blend_weights['Hándicap']['ml_weight']))
                    filas=[]
                    for cand in candidatos_ind:
                        if cand is None:continue
                        filas.append({'Selección':cand.selection,'Mercado':cand.market,'Prob. ML':f'{cand.prob_ml:.1f}%','Prob. MC':f'{cand.prob_mc:.1f}%','Prob. Combinada':f'{cand.probability:.1f}%','Peso ML':f'{cand.blend_weight_ml*100:.0f}%','Push':f'{cand.push_probability:.1f}%','Cuota':cand.odds,'Edge':None if cand.edge_pp is None else f'{cand.edge_pp:.2f} pp','EV':f'{cand.ev_pct:.2f}%','Kelly':f'{calcular_criterio_kelly(cand.probability,cand.odds,prob_push=cand.push_probability)}%' if cand.accepted else '0.0%','Estado':'✅ CANDIDATO' if cand.accepted else 'NO BET','Motivo':cand.reason})
                    st.subheader('3. Veredicto Financiero Unificado');st.dataframe(pd.DataFrame(filas),use_container_width=True,hide_index=True)
