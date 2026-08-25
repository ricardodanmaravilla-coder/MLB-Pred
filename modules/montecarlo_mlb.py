import hashlib
import numpy as np
import pandas as pd
from .team_utils import normalize_team
from .historical_mlb import prepare_games


def calcular_factor_clima(viento_mph,direccion_viento,temp_f):
    try: wind=float(viento_mph or 0); temp=float(temp_f or 72)
    except (TypeError,ValueError): wind,temp=0.,72.
    d=str(direccion_viento or '').lower(); mult=1.
    if 'outfield' in d or 'hacia afuera' in d: mult+=min(wind,25.)*.004
    elif 'infield' in d or 'hacia adentro' in d: mult-=min(wind,25.)*.004
    mult+=np.clip(temp-72.,-30.,30.)*.0015; return float(np.clip(mult,.88,1.12))
def _games(df):
    g=prepare_games(df)
    if not g.empty:
        g=g.copy(); g['HomeKey']=g.Home.map(normalize_team); g['AwayKey']=g.Away.map(normalize_team)
    return g
def obtener_h2h_detalle(df_games,loc_abbr,vis_abbr,n=12):
    g=_games(df_games)
    if g.empty: return 50.,0
    loc,vis=normalize_team(loc_abbr),normalize_team(vis_abbr); rows=g[((g.HomeKey==loc)&(g.AwayKey==vis))|((g.HomeKey==vis)&(g.AwayKey==loc))].tail(n); wins=valid=0
    for _,r in rows.iterrows():
        hs,aw=float(r.Home_Score),float(r.Away_Score)
        if hs==aw: continue
        valid+=1
        if (r.HomeKey==loc and hs>aw) or (r.AwayKey==loc and aw>hs): wins+=1
    if not valid: return 50.,0
    raw=wins/valid; weight=valid/(valid+10.); return float((.5+(raw-.5)*weight)*100.),valid
def obtener_h2h(df_games,loc_abbr,vis_abbr): return obtener_h2h_detalle(df_games,loc_abbr,vis_abbr)[0]
def obtener_carreras_recientes(df_games,equipo_abbr,n=10):
    g=_games(df_games)
    if g.empty: return None
    t=normalize_team(equipo_abbr); rows=g[(g.HomeKey==t)|(g.AwayKey==t)].tail(int(n)); vals=[float(r.Home_Score if r.HomeKey==t else r.Away_Score) for _,r in rows.iterrows()]
    return float(np.mean(vals)) if vals else None
def _spread_probability(diff,side,line):
    margin=diff+float(line) if side=='local' else -diff+float(line); return float(np.mean(margin>0)*100.)
def _spread_push_probability(diff,side,line):
    margin=diff+float(line) if side=='local' else -diff+float(line); return float(np.mean(np.isclose(margin,0.))*100.)
def _resolve_extra_innings(home,away,tied_mask,exp_home,exp_away,rng):
    idx=np.flatnonzero(tied_mask)
    if len(idx)==0: return home,away,0
    lam_h=max(.25,min(1.25,(float(exp_home)/9.)*1.35)); lam_a=max(.25,min(1.25,(float(exp_away)/9.)*1.35)); active=idx.copy(); innings_used=0
    for inning in range(1,7):
        if len(active)==0: break
        innings_used=inning; away[active]+=rng.poisson(lam_a,len(active)); home[active]+=rng.poisson(lam_h*1.02,len(active)); active=active[home[active]==away[active]]
    if len(active):
        hw=rng.random(len(active))<.52; home[active]+=hw.astype(int); away[active]+=(~hw).astype(int)
    return home,away,innings_used
def _stable_seed(parts): return int.from_bytes(hashlib.sha256('|'.join(str(x) for x in parts).encode('utf-8',errors='ignore')).digest()[:8],'big')%(2**32-1)
def _starter_share(expected_ip):
    """Translate expected starter innings to share of regulation pitching workload."""
    try: ip=float(expected_ip)
    except (TypeError,ValueError): ip=5.2
    return float(np.clip(ip/9.0,.45,.72))


def simular_partido_mlb(local,visita,pitcher_loc_xfip,pitcher_vis_xfip,wrc_loc,wrc_vis,bullpen_loc_era,bullpen_vis_era,park_factor,altitud_ft,viento_mph,direccion_viento,temp_f,linea_carreras_casino,df_games=None,num_simulaciones=50000,simulation_seed=None,starter_ip_loc=None,starter_ip_vis=None):
    if linea_carreras_casino is None or float(linea_carreras_casino)<=0: raise ValueError('Línea de carreras de casino requerida y no disponible.')
    vals=[wrc_loc,wrc_vis,pitcher_loc_xfip,pitcher_vis_xfip,bullpen_loc_era,bullpen_vis_era,park_factor]
    if any(v is None or pd.isna(v) for v in vals): raise ValueError(f'Datos incompletos para simular {visita} @ {local}.')
    loc,vis=normalize_team(local),normalize_team(visita); bat_l=np.clip(float(wrc_loc)/100.,.75,1.25); bat_v=np.clip(float(wrc_vis)/100.,.75,1.25)
    sp_l=np.clip(float(pitcher_loc_xfip)/4.10,.70,1.30); sp_v=np.clip(float(pitcher_vis_xfip)/4.10,.70,1.30); team_pitch_l=np.clip(float(bullpen_loc_era)/4.10,.75,1.30); team_pitch_v=np.clip(float(bullpen_vis_era)/4.10,.75,1.30)
    climate=calcular_factor_clima(viento_mph,direccion_viento,temp_f); park=np.clip(float(park_factor)/100.,.88,1.12)
    try: altitude=1.+float(np.clip((float(altitud_ft or 0)-1000.)/1000.*.0015,0.,.008))
    except (TypeError,ValueError): altitude=1.
    share_l=_starter_share(starter_ip_loc); share_v=_starter_share(starter_ip_vis)
    # Home offense faces visitor starter/bullpen; away offense faces home starter/bullpen.
    opp_pitch_home=sp_v*share_v+team_pitch_v*(1-share_v); opp_pitch_away=sp_l*share_l+team_pitch_l*(1-share_l)
    exp_l=float(np.clip(4.55*bat_l*opp_pitch_home*park*climate*altitude,1.5,8.5)); exp_v=float(np.clip(4.45*bat_v*opp_pitch_away*park*climate*altitude,1.5,8.5))
    sims=int(np.clip(num_simulaciones,5000,100000))
    if simulation_seed is None:
        simulation_seed=_stable_seed([loc,vis,round(float(pitcher_loc_xfip),3),round(float(pitcher_vis_xfip),3),round(float(wrc_loc),3),round(float(wrc_vis),3),round(float(bullpen_loc_era),3),round(float(bullpen_vis_era),3),round(float(park_factor),3),round(float(altitud_ft or 0),1),round(float(viento_mph or 0),1),str(direccion_viento),round(float(temp_f or 72),1),round(float(linea_carreras_casino),2),round(share_l,3),round(share_v,3),sims])
    rng=np.random.default_rng(int(simulation_seed)); dispersion=14.5; home=rng.negative_binomial(dispersion,dispersion/(dispersion+exp_l),sims); away=rng.negative_binomial(dispersion,dispersion/(dispersion+exp_v),sims)
    ties=home==away; home,away,max_extra=_resolve_extra_innings(home,away,ties,exp_l,exp_v,rng); prob_l=float(np.mean(home>away)*100.); prob_v=100.-prob_l; totals=home+away; diff=home-away; line=float(linea_carreras_casino)
    runs={'Promedio_Total':round(float(np.mean(totals)),2),f'Over {linea_carreras_casino}':round(float(np.mean(totals>line)*100.),2),f'Under {linea_carreras_casino}':round(float(np.mean(totals<line)*100.),2),f'Push {linea_carreras_casino}':round(float(np.mean(np.isclose(totals,line))*100.),2)}
    for spread in np.arange(-5.5,6.0,.5):
        if abs(spread)<1e-9: continue
        spread=float(spread); runs[f'Spread Local {spread:+.1f}']=round(_spread_probability(diff,'local',spread),2); runs[f'Spread Visita {spread:+.1f}']=round(_spread_probability(diff,'visita',spread),2); runs[f'Push Spread Local {spread:+.1f}']=round(_spread_push_probability(diff,'local',spread),2); runs[f'Push Spread Visita {spread:+.1f}']=round(_spread_push_probability(diff,'visita',spread),2)
    h2h,h2h_n=obtener_h2h_detalle(df_games,loc,vis); pexp=(exp_l+exp_v)**.285; pyth=(exp_l**pexp)/((exp_l**pexp)+(exp_v**pexp))*100.
    return {'Moneyline':{'Gana Local':round(prob_l,2),'Gana Visita':round(prob_v,2)},'Carreras':runs,'Metadatos':{'Pythagenpat_Loc':round(float(pyth),2),'H2H_Loc':round(h2h,2),'H2H_Muestra':h2h_n,'H2H_Peso':0.,'H2H_Usado_En_Probabilidad':False,'Forma_Reciente_Usada_En_MC':False,'Factor_Clima':round(climate,4),'Factor_Altitud_Residual':round(altitude,4),'Carreras_Exp_Local':round(exp_l,2),'Carreras_Exp_Visita':round(exp_v,2),'StarterShare_Local':round(share_l,3),'StarterShare_Visita':round(share_v,3),'Empates_Regulacion':int(ties.sum()),'Max_Extra_Innings_Simulados':int(max_extra),'Pitching_Agregado_Es_Proxy_Bullpen':True,'Simulaciones':sims,'Simulation_Seed':int(simulation_seed)}}
