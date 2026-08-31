"""Validate Moneyline edge/EV filters from archived real pregame odds.

This is deliberately prospective: it never fabricates historical prices.  It joins
real snapshots to leak-safe model predictions and settled MLB results.  Until the
archive is large enough it returns COLLECTING_DATA, never a promotion.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from modules.team_utils import normalize_team

ODDS=Path('data/mlb_odds_history.csv')
PREDS=Path('artifacts/ml_mc_blend_walkforward_predictions.csv')
GAMES=Path('data/mlb_games.csv')
OUT=Path('artifacts/edge_ev_validation.json')
CURRENT_EDGE=4.0; CURRENT_EV=4.0
MIN_BASELINE_BETS=150

def _num(x): return pd.to_numeric(x,errors='coerce')
def _novig(a,b):
    ia,ib=1.0/a,1.0/b; s=ia+ib
    return ia/s,ib/s

def _metrics(d):
    if d.empty:return {'bets':0,'hit_rate':None,'roi':None,'units':0.0,'avg_edge_pp':None,'avg_ev_pct':None}
    units=np.where(d.win.eq(1),d.odds_decimal-1.0,-1.0)
    return {'bets':int(len(d)),'hit_rate':float(d.win.mean()),'roi':float(units.mean()),'units':float(units.sum()),'avg_edge_pp':float(d.edge_pp.mean()),'avg_ev_pct':float(d.ev_pct.mean())}

def main():
    OUT.parent.mkdir(exist_ok=True)
    if not ODDS.exists() or ODDS.stat().st_size<120:
        result={'decision':'COLLECTING_DATA','reason':'sin_historial_de_cuotas_reales','production_changed':False}
        OUT.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));return
    o=pd.read_csv(ODDS)
    if o.empty:
        result={'decision':'COLLECTING_DATA','reason':'historial_vacio','production_changed':False};OUT.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));return
    o=o[o.market.astype(str).eq('h2h')].copy(); o['snapshot_utc']=pd.to_datetime(o.snapshot_utc,utc=True,errors='coerce');o['commence']=pd.to_datetime(o.commence_time_utc,utc=True,errors='coerce')
    o=o.dropna(subset=['snapshot_utc','commence']);o=o[(o.snapshot_utc<=o.commence)&((o.commence-o.snapshot_utc)<=pd.Timedelta(hours=6))]
    if o.empty:
        result={'decision':'COLLECTING_DATA','reason':'sin_snapshots_pregame_validos','production_changed':False};OUT.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));return
    o['home_key']=o.home.map(normalize_team);o['away_key']=o.away.map(normalize_team);o['sel_key']=o.selection.map(normalize_team);o['odds_decimal']=_num(o.odds_decimal)
    # Closing-ish: latest available snapshot for each book/side before first pitch.
    o=o.sort_values('snapshot_utc').groupby(['event_id','book','sel_key'],as_index=False).tail(1)
    rec=[]
    for (eid,book),g in o.groupby(['event_id','book']):
        h=str(g.home_key.iloc[0]);a=str(g.away_key.iloc[0]);comm=g.commence.iloc[0]
        hr=g[g.sel_key.eq(h)];ar=g[g.sel_key.eq(a)]
        if hr.empty or ar.empty:continue
        ho=float(hr.odds_decimal.iloc[-1]);ao=float(ar.odds_decimal.iloc[-1])
        if ho<=1 or ao<=1:continue
        nh,na=_novig(ho,ao)
        rec.append({'event_id':eid,'book':book,'Date':comm.tz_convert('America/Chicago').date().isoformat(),'home_key':h,'away_key':a,'home_odds':ho,'away_odds':ao,'home_nv':nh,'away_nv':na})
    b=pd.DataFrame(rec)
    if b.empty:
        result={'decision':'COLLECTING_DATA','reason':'sin_moneylines_dos_vias','production_changed':False};OUT.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));return
    # Median across books approximates the consensus actually used by the multi-source layer.
    c=b.groupby(['Date','home_key','away_key'],as_index=False).agg(home_odds=('home_odds','median'),away_odds=('away_odds','median'),home_nv=('home_nv','median'),away_nv=('away_nv','median'),books=('book','nunique'))
    if not PREDS.exists():
        result={'decision':'COLLECTING_DATA','reason':'faltan_predicciones_walkforward','odds_games':int(len(c)),'production_changed':False};OUT.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));return
    p=pd.read_csv(PREDS);p['GameID']=_num(p.GameID);p['Date']=pd.to_datetime(p.Date,errors='coerce').dt.date.astype(str)
    g=pd.read_csv(GAMES);g['GameID']=_num(g.get('GameID'));g['Date']=pd.to_datetime(g.Date,errors='coerce').dt.date.astype(str);g['home_key']=g.Home.map(normalize_team);g['away_key']=g.Away.map(normalize_team)
    gp=g[['GameID','Date','home_key','away_key','Home_Score','Away_Score']].dropna(subset=['GameID']).drop_duplicates('GameID')
    p=p.merge(gp,on=['GameID','Date'],how='left').merge(c,on=['Date','home_key','away_key'],how='inner')
    rows=[]
    for _,r in p.iterrows():
        phml=float(r.prob_ml);phmc=float(r.prob_mc);ph=.5*(phml+phmc)
        for side,pml,pmc,prob,nv,odds,win in [
            ('home',phml,phmc,ph,float(r.home_nv),float(r.home_odds),int(float(r.Home_Score)>float(r.Away_Score))),
            ('away',1-phml,1-phmc,1-ph,float(r.away_nv),float(r.away_odds),int(float(r.Away_Score)>float(r.Home_Score)))]:
            if pml<.58 or pmc<.58 or prob<.58 or abs(pml-pmc)>.10:continue
            edge=(prob-nv)*100.;ev=(prob*(odds-1)-(1-prob))*100.
            rows.append({'Date':r.Date,'GameID':int(r.GameID),'side':side,'win':win,'odds_decimal':odds,'edge_pp':edge,'ev_pct':ev,'probability':prob,'books':int(r.books)})
    d=pd.DataFrame(rows)
    if d.empty:
        result={'decision':'COLLECTING_DATA','reason':'sin_apuestas_que_pasaron_filtro_deportivo','matched_games':int(len(p)),'production_changed':False};OUT.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));return
    base=d[(d.edge_pp>=CURRENT_EDGE)&(d.ev_pct>=CURRENT_EV)].copy(); bm=_metrics(base)
    result={'decision':'COLLECTING_DATA','production_changed':False,'real_odds_games':int(len(c)),'matched_model_games':int(len(p)),'sports_filter_candidates':int(len(d)),'current_thresholds':{'edge_pp':CURRENT_EDGE,'ev_pct':CURRENT_EV},'current_results':bm,'minimum_baseline_bets_for_validation':MIN_BASELINE_BETS}
    if len(base)>=MIN_BASELINE_BETS:
        trials=[]
        for edge in (2.,3.,4.,5.,6.,7.,8.):
            for ev in (2.,3.,4.,5.,6.,8.,10.):
                q=d[(d.edge_pp>=edge)&(d.ev_pct>=ev)];m=_metrics(q)
                if m['bets']>=max(60,int(.70*len(base))):trials.append({'edge_pp':edge,'ev_pct':ev,**m})
        trials.sort(key=lambda z:(z['roi'] if z['roi'] is not None else -9,z['hit_rate'] if z['hit_rate'] is not None else -9),reverse=True)
        result['diagnostic_trials_not_promoted_automatically']=trials[:10]
        result['decision']='READY_FOR_STRICT_TEMPORAL_GATE'
    OUT.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
