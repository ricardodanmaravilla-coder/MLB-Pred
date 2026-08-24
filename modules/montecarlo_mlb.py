import numpy as np
import pandas as pd
from .team_utils import normalize_team
from .historical_mlb import prepare_games


def calcular_factor_clima(viento_mph,direccion_viento,temp_f):
    try: wind=float(viento_mph or 0); temp=float(temp_f or 72)
    except (TypeError,ValueError): wind,temp=0.,72.
    d=str(direccion_viento or '').lower(); mult=1.0
    if 'outfield' in d or 'hacia afuera' in d: mult += min(wind,25.)*.004
    elif 'infield' in d or 'hacia adentro' in d: mult -= min(wind,25.)*.004
    mult += np.clip(temp-72.,-30.,30.)*.0015
    return float(np.clip(mult,.88,1.12))


def _games(df):
    g=prepare_games(df)
    if not g.empty:
        g=g.copy(); g['HomeKey']=g.Home.map(normalize_team); g['AwayKey']=g.Away.map(normalize_team)
    return g


def obtener_h2h_detalle(df_games,loc_abbr,vis_abbr,n=12):
    g=_games(df_games)
    if g.empty: return 50.,0
    loc,vis=normalize_team(loc_abbr),normalize_team(vis_abbr)
    rows=g[((g.HomeKey==loc)&(g.AwayKey==vis))|((g.HomeKey==vis)&(g.AwayKey==loc))].tail(n)
    wins=valid=0
    for _,r in rows.iterrows():
        hs,as_=float(r.Home_Score),float(r.Away_Score)
        if hs==as_: continue
        valid+=1
        if (r.HomeKey==loc and hs>as_) or (r.AwayKey==loc and as_>hs): wins+=1
    if not valid:return 50.,0
    raw=wins/valid; weight=valid/(valid+10.0); shrunk=.5+(raw-.5)*weight
    return float(shrunk*100.),valid


def obtener_h2h(df_games,loc_abbr,vis_abbr): return obtener_h2h_detalle(df_games,loc_abbr,vis_abbr)[0]


def obtener_carreras_recientes(df_games,equipo_abbr,n=10):
    g=_games(df_games)
    if g.empty:return None
    t=normalize_team(equipo_abbr); rows=g[(g.HomeKey==t)|(g.AwayKey==t)].tail(int(n)); vals=[]
    for _,r in rows.iterrows(): vals.append(float(r.Home_Score if r.HomeKey==t else r.Away_Score))
    return float(np.mean(vals)) if vals else None


def _spread_probability(diff,side,line):
    line=float(line); return float(np.mean((diff+line)>0)*100.) if side=='local' else float(np.mean((-diff+line)>0)*100.)


def simular_partido_mlb(local,visita,pitcher_loc_xfip,pitcher_vis_xfip,wrc_loc,wrc_vis,bullpen_loc_era,bullpen_vis_era,park_factor,altitud_ft,viento_mph,direccion_viento,temp_f,linea_carreras_casino,df_games=None,num_simulaciones=50000):
    if linea_carreras_casino is None or float(linea_carreras_casino)<=0: raise ValueError('Línea de carreras de casino requerida y no disponible.')
    vals=[wrc_loc,wrc_vis,pitcher_loc_xfip,pitcher_vis_xfip,bullpen_loc_era,bullpen_vis_era,park_factor]
    if any(v is None or pd.isna(v) for v in vals): raise ValueError(f'Datos incompletos para simular {visita} @ {local}.')
    loc,vis=normalize_team(local),normalize_team(visita); rl=obtener_carreras_recientes(df_games,loc); rv=obtener_carreras_recientes(df_games,vis)
    batl=np.clip(float(wrc_loc)/100.,.75,1.25); batv=np.clip(float(wrc_vis)/100.,.75,1.25)
    spl=np.clip(float(pitcher_loc_xfip)/4.10,.70,1.30); spv=np.clip(float(pitcher_vis_xfip)/4.10,.70,1.30)
    bpl=np.clip(float(bullpen_loc_era)/4.10,.75,1.30); bpv=np.clip(float(bullpen_vis_era)/4.10,.75,1.30)
    climate=calcular_factor_clima(viento_mph,direccion_viento,temp_f); park=np.clip(float(park_factor)/100.,.88,1.12)
    # Park factor already captures most altitude. Add only a small residual, capped at 0.8%, to avoid double counting Coors.
    try: altitude=1.0+float(np.clip((float(altitud_ft or 0)-1000.)/1000.*.0015,0.,.008))
    except (TypeError,ValueError): altitude=1.0
    base_l=4.55*batl*(spv*.60+bpv*.40)*park*climate*altitude; base_v=4.45*batv*(spl*.60+bpl*.40)*park*climate*altitude
    exp_l=base_l if rl is None else .70*base_l+.30*rl; exp_v=base_v if rv is None else .70*base_v+.30*rv
    exp_l=float(np.clip(exp_l,1.5,8.5)); exp_v=float(np.clip(exp_v,1.5,8.5)); sims=int(np.clip(num_simulaciones,5000,100000)); rng=np.random.default_rng(); disp=14.5
    cl=rng.negative_binomial(disp,disp/(disp+exp_l),sims); cv=rng.negative_binomial(disp,disp/(disp+exp_v),sims)
    ties=cl==cv; nt=int(ties.sum())
    if nt:
        lw=rng.random(nt)<.53; cl[ties]+=lw.astype(int); cv[ties]+=(~lw).astype(int)
    mc=float(np.mean(cl>cv)*100.); h2h,hn=obtener_h2h_detalle(df_games,loc,vis)
    # Historical matchup has deliberately small influence and is shrinkage-adjusted.
    h_weight=min(.10,hn*.01); prob_l=(1-h_weight)*mc+h_weight*h2h; prob_v=100.-prob_l
    totals=cl+cv; diff=cl-cv; line=float(linea_carreras_casino)
    runs={'Promedio_Total':round(float(np.mean(totals)),2),f'Over {linea_carreras_casino}':round(float(np.mean(totals>line)*100),2),f'Under {linea_carreras_casino}':round(float(np.mean(totals<line)*100),2),f'Push {linea_carreras_casino}':round(float(np.mean(totals==line)*100),2)}
    for s in np.arange(-5.5,6.,.5):
        if abs(s)<1e-9:continue
        s=float(s); runs[f'Spread Local {s:+.1f}']=round(_spread_probability(diff,'local',s),2); runs[f'Spread Visita {s:+.1f}']=round(_spread_probability(diff,'visita',s),2)
    pexp=(exp_l+exp_v)**.285; pyth=(exp_l**pexp)/((exp_l**pexp)+(exp_v**pexp))*100.
    return {'Moneyline':{'Gana Local':round(prob_l,2),'Gana Visita':round(prob_v,2)},'Carreras':runs,'Metadatos':{'Pythagenpat_Loc':round(float(pyth),2),'H2H_Loc':round(h2h,2),'H2H_Muestra':hn,'H2H_Peso':round(h_weight,3),'H2H_Usado_En_Probabilidad':True,'Factor_Clima':round(climate,4),'Factor_Altitud_Residual':round(altitude,4),'Carreras_Exp_Local':round(exp_l,2),'Carreras_Exp_Visita':round(exp_v,2),'Simulaciones':sims}}
