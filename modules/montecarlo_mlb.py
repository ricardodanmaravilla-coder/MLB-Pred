import hashlib

import numpy as np
import pandas as pd

from .team_utils import normalize_team
from .historical_mlb import prepare_games


def calcular_factor_clima(viento_mph, direccion_viento, temp_f):
    try:
        wind=float(viento_mph or 0); temp=float(temp_f or 72)
    except (TypeError,ValueError): wind,temp=0.0,72.0
    d=str(direccion_viento or '').lower(); mult=1.0
    if 'outfield' in d or 'hacia afuera' in d: mult+=min(wind,25.0)*0.004
    elif 'infield' in d or 'hacia adentro' in d: mult-=min(wind,25.0)*0.004
    mult+=np.clip(temp-72.0,-30.0,30.0)*0.0015
    return float(np.clip(mult,0.88,1.12))


def _games(df):
    g=prepare_games(df)
    if not g.empty:
        g=g.copy(); g['HomeKey']=g.Home.map(normalize_team); g['AwayKey']=g.Away.map(normalize_team); g['Date']=pd.to_datetime(g['Date'],errors='coerce')
    return g


def _estimate_dispersion(df_games,fallback=14.5):
    try:
        g=prepare_games(df_games)
        if g.empty or len(g)<300:return float(fallback),'fallback_sparse'
        g=g.sort_values('Date').tail(2500); scores=pd.concat([pd.to_numeric(g['Home_Score'],errors='coerce'),pd.to_numeric(g['Away_Score'],errors='coerce')],ignore_index=True).dropna().astype(float); scores=scores[(scores>=0)&(scores<=30)]
        if len(scores)<600:return float(fallback),'fallback_sparse'
        mu=float(scores.mean());var=float(scores.var(ddof=1));excess=var-mu
        if not np.isfinite(mu) or not np.isfinite(var) or mu<=0 or excess<=0.05:return float(fallback),'fallback_invalid'
        return float(np.clip((mu*mu)/excess,3.0,30.0)),'historical_moments'
    except Exception:return float(fallback),'fallback_error'


def _team_schedule_load(df_games,team):
    """Conservative bullpen fatigue proxy from completed schedule only.

    Returns a multiplier for run-prevention difficulty: >1 means a slightly more
    fatigued staff. The adjustment is deliberately capped at +/-3.5% because game
    results do not reveal actual reliever pitch counts.
    """
    try:
        g=_games(df_games); t=normalize_team(team)
        if g.empty:return 1.0,3,0,'neutral_no_history'
        rows=g[(g.HomeKey==t)|(g.AwayKey==t)].dropna(subset=['Date']).sort_values('Date')
        if rows.empty:return 1.0,3,0,'neutral_no_team_history'
        latest_all=g['Date'].dropna().max().normalize(); today=pd.Timestamp.now(tz='America/New_York').tz_localize(None).normalize()
        as_of=today if abs((today-latest_all).days)<=3 else latest_all+pd.Timedelta(days=1)
        prior=rows[rows['Date'].dt.normalize()<as_of]
        if prior.empty:return 1.0,3,0,'neutral_no_prior'
        last=prior['Date'].max().normalize(); rest=max(0,min(7,(as_of-last).days-1)); recent=int((prior['Date'].dt.normalize()>=as_of-pd.Timedelta(days=3)).sum())
        adjustment=0.0
        if rest==0: adjustment+=0.012
        elif rest>=2: adjustment-=0.008
        adjustment+=max(0,recent-2)*0.008
        factor=float(np.clip(1.0+adjustment,0.975,1.035))
        return factor,int(rest),recent,'completed_schedule_proxy'
    except Exception:return 1.0,3,0,'neutral_error'


def obtener_h2h_detalle(df_games,loc_abbr,vis_abbr,n=12):
    g=_games(df_games)
    if g.empty:return 50.0,0
    loc,vis=normalize_team(loc_abbr),normalize_team(vis_abbr);rows=g[((g.HomeKey==loc)&(g.AwayKey==vis))|((g.HomeKey==vis)&(g.AwayKey==loc))].tail(n);wins=valid=0
    for _,r in rows.iterrows():
        hs,aw=float(r.Home_Score),float(r.Away_Score)
        if hs==aw:continue
        valid+=1
        if (r.HomeKey==loc and hs>aw) or (r.AwayKey==loc and aw>hs):wins+=1
    if not valid:return 50.0,0
    raw=wins/valid;weight=valid/(valid+10.0);return float((0.5+(raw-0.5)*weight)*100.0),valid


def obtener_h2h(df_games,loc_abbr,vis_abbr):return obtener_h2h_detalle(df_games,loc_abbr,vis_abbr)[0]

def obtener_carreras_recientes(df_games,equipo_abbr,n=10):
    g=_games(df_games)
    if g.empty:return None
    t=normalize_team(equipo_abbr);rows=g[(g.HomeKey==t)|(g.AwayKey==t)].tail(int(n));vals=[float(r.Home_Score if r.HomeKey==t else r.Away_Score) for _,r in rows.iterrows()]
    return float(np.mean(vals)) if vals else None

def _spread_probability(diff,side,line):
    margin=diff+float(line) if side=='local' else -diff+float(line);return float(np.mean(margin>0)*100.0)

def _spread_push_probability(diff,side,line):
    margin=diff+float(line) if side=='local' else -diff+float(line);return float(np.mean(np.isclose(margin,0.0))*100.0)

def _resolve_extra_innings(home,away,tied_mask,exp_home,exp_away,rng):
    idx=np.flatnonzero(tied_mask)
    if len(idx)==0:return home,away,0
    lam_h=max(0.25,min(1.25,(float(exp_home)/9.0)*1.35));lam_a=max(0.25,min(1.25,(float(exp_away)/9.0)*1.35));active=idx.copy();innings_used=0
    for inning in range(1,7):
        if len(active)==0:break
        innings_used=inning;add_a=rng.poisson(lam_a,len(active));add_h=rng.poisson(lam_h*1.02,len(active));away[active]+=add_a;home[active]+=add_h;active=active[home[active]==away[active]]
    if len(active):
        home_win=rng.random(len(active))<0.52;home[active]+=home_win.astype(int);away[active]+=(~home_win).astype(int)
    return home,away,innings_used

def _stable_seed(parts):
    raw='|'.join(str(x) for x in parts).encode('utf-8',errors='ignore');return int.from_bytes(hashlib.sha256(raw).digest()[:8],'big')%(2**32-1)


def simular_partido_mlb(local,visita,pitcher_loc_xfip,pitcher_vis_xfip,wrc_loc,wrc_vis,bullpen_loc_era,bullpen_vis_era,park_factor,altitud_ft,viento_mph,direccion_viento,temp_f,linea_carreras_casino,df_games=None,num_simulaciones=50000,simulation_seed=None):
    if linea_carreras_casino is None or float(linea_carreras_casino)<=0:raise ValueError('Línea de carreras de casino requerida y no disponible.')
    vals=[wrc_loc,wrc_vis,pitcher_loc_xfip,pitcher_vis_xfip,bullpen_loc_era,bullpen_vis_era,park_factor]
    if any(v is None or pd.isna(v) for v in vals):raise ValueError(f'Datos incompletos para simular {visita} @ {local}.')
    loc,vis=normalize_team(local),normalize_team(visita);bat_l=np.clip(float(wrc_loc)/100.0,0.75,1.25);bat_v=np.clip(float(wrc_vis)/100.0,0.75,1.25);sp_l=np.clip(float(pitcher_loc_xfip)/4.10,0.70,1.30);sp_v=np.clip(float(pitcher_vis_xfip)/4.10,0.70,1.30);team_pitch_l=np.clip(float(bullpen_loc_era)/4.10,0.75,1.30);team_pitch_v=np.clip(float(bullpen_vis_era)/4.10,0.75,1.30)
    fatigue_l,rest_l,recent_l,fatigue_src_l=_team_schedule_load(df_games,loc);fatigue_v,rest_v,recent_v,fatigue_src_v=_team_schedule_load(df_games,vis);team_pitch_l*=fatigue_l;team_pitch_v*=fatigue_v
    climate=calcular_factor_clima(viento_mph,direccion_viento,temp_f);park=np.clip(float(park_factor)/100.0,0.88,1.12)
    try:altitude=1.0+float(np.clip((float(altitud_ft or 0)-1000.0)/1000.0*0.0015,0.0,0.008))
    except (TypeError,ValueError):altitude=1.0
    opp_pitch_for_home=sp_v*0.72+team_pitch_v*0.28;opp_pitch_for_away=sp_l*0.72+team_pitch_l*0.28;exp_l=4.55*bat_l*opp_pitch_for_home*park*climate*altitude;exp_v=4.45*bat_v*opp_pitch_for_away*park*climate*altitude;exp_l=float(np.clip(exp_l,1.5,8.5));exp_v=float(np.clip(exp_v,1.5,8.5))
    sims=int(np.clip(num_simulaciones,5000,100000));dispersion,dispersion_source=_estimate_dispersion(df_games,14.5)
    if simulation_seed is None:
        simulation_seed=_stable_seed([loc,vis,round(float(pitcher_loc_xfip),3),round(float(pitcher_vis_xfip),3),round(float(wrc_loc),3),round(float(wrc_vis),3),round(float(bullpen_loc_era),3),round(float(bullpen_vis_era),3),round(float(fatigue_l),4),round(float(fatigue_v),4),round(float(park_factor),3),round(float(altitud_ft or 0),1),round(float(viento_mph or 0),1),str(direccion_viento),round(float(temp_f or 72),1),round(float(linea_carreras_casino),2),round(float(dispersion),4),sims])
    rng=np.random.default_rng(int(simulation_seed));home=rng.negative_binomial(dispersion,dispersion/(dispersion+exp_l),sims);away=rng.negative_binomial(dispersion,dispersion/(dispersion+exp_v),sims);regulation_ties=home==away;home,away,max_extra_innings=_resolve_extra_innings(home,away,regulation_ties,exp_l,exp_v,rng)
    prob_l=float(np.mean(home>away)*100.0);prob_v=100.0-prob_l;totals=home+away;diff=home-away;line=float(linea_carreras_casino)
    runs={'Promedio_Total':round(float(np.mean(totals)),2),f'Over {linea_carreras_casino}':round(float(np.mean(totals>line)*100.0),2),f'Under {linea_carreras_casino}':round(float(np.mean(totals<line)*100.0),2),f'Push {linea_carreras_casino}':round(float(np.mean(np.isclose(totals,line))*100.0),2)}
    for spread in np.arange(-5.5,6.0,0.5):
        if abs(spread)<1e-9:continue
        spread=float(spread);runs[f'Spread Local {spread:+.1f}']=round(_spread_probability(diff,'local',spread),2);runs[f'Spread Visita {spread:+.1f}']=round(_spread_probability(diff,'visita',spread),2);runs[f'Push Spread Local {spread:+.1f}']=round(_spread_push_probability(diff,'local',spread),2);runs[f'Push Spread Visita {spread:+.1f}']=round(_spread_push_probability(diff,'visita',spread),2)
    h2h,h2h_n=obtener_h2h_detalle(df_games,loc,vis);pexp=(exp_l+exp_v)**0.285;pyth=(exp_l**pexp)/((exp_l**pexp)+(exp_v**pexp))*100.0
    return {'Moneyline':{'Gana Local':round(prob_l,2),'Gana Visita':round(prob_v,2)},'Carreras':runs,'Metadatos':{'Pythagenpat_Loc':round(float(pyth),2),'H2H_Loc':round(h2h,2),'H2H_Muestra':h2h_n,'H2H_Peso':0.0,'H2H_Usado_En_Probabilidad':False,'Forma_Reciente_Usada_En_MC':False,'Factor_Clima':round(climate,4),'Factor_Altitud_Residual':round(altitude,4),'Carreras_Exp_Local':round(exp_l,2),'Carreras_Exp_Visita':round(exp_v,2),'Bullpen_Load_Factor_Local':round(fatigue_l,4),'Bullpen_Load_Factor_Visita':round(fatigue_v,4),'Rest_Days_Local':rest_l,'Rest_Days_Visita':rest_v,'Games_Last3_Local':recent_l,'Games_Last3_Visita':recent_v,'Bullpen_Load_Source_Local':fatigue_src_l,'Bullpen_Load_Source_Visita':fatigue_src_v,'Dispersion_NB':round(float(dispersion),4),'Dispersion_Source':dispersion_source,'Empates_Regulacion':int(regulation_ties.sum()),'Max_Extra_Innings_Simulados':int(max_extra_innings),'Pitching_Agregado_Es_Proxy_Bullpen':True,'Simulaciones':sims,'Simulation_Seed':int(simulation_seed)}}
