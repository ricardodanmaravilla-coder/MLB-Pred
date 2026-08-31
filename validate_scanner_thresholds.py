"""Validation-only temporal optimization of sports-signal scanner thresholds.

This experiment deliberately excludes edge/EV because the repository does not contain
historical sportsbook odds for the walk-forward sample. It tests only the production
moneyline sports filters: minimum ML probability, minimum MC probability, minimum
combined probability, and maximum ML-MC disagreement. Production is never changed.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

SRC=Path('artifacts/ml_mc_blend_walkforward_predictions.csv')
OUT=Path('artifacts/scanner_threshold_predictions.csv')
REPORT=Path('artifacts/scanner_threshold_report.json')
CURRENT={'min_ml':58.0,'min_mc':58.0,'min_combined':58.0,'max_disagreement':10.0}

GRID=[]
for min_ml in (56.,58.,60.,62.):
  for min_mc in (56.,58.,60.,62.):
    for min_combined in (56.,58.,60.,62.,64.):
      for max_disagreement in (6.,8.,10.,12.):
        if min_combined < min(min_ml,min_mc)-2: continue
        GRID.append({'min_ml':min_ml,'min_mc':min_mc,'min_combined':min_combined,'max_disagreement':max_disagreement})

def _candidate_rows(df,cfg):
    x=df.copy(); ml=x.prob_ml.to_numpy(float); mc=x.prob_mc.to_numpy(float); comb=(ml+mc)/2
    home=(ml>=cfg['min_ml']/100)&(mc>=cfg['min_mc']/100)&(comb>=cfg['min_combined']/100)&(np.abs(ml-mc)<=cfg['max_disagreement']/100)
    aml=1-ml; amc=1-mc; acom=(aml+amc)/2
    away=(aml>=cfg['min_ml']/100)&(amc>=cfg['min_mc']/100)&(acom>=cfg['min_combined']/100)&(np.abs(aml-amc)<=cfg['max_disagreement']/100)
    side=np.where(home,1,np.where(away,0,-1)); mask=side>=0
    g=x.loc[mask].copy(); g['pick_home']=side[mask]; g['correct']=(g.pick_home.to_numpy(int)==g.actual_home_win.to_numpy(int)).astype(int)
    g['pick_probability']=np.where(g.pick_home.to_numpy(int)==1,comb[mask],1-comb[mask])
    return g

def _wilson_lower(wins,n,z=1.96):
    if n<=0:return 0.0
    p=wins/n; den=1+z*z/n; ctr=p+z*z/(2*n); adj=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)
    return (ctr-adj)/den

def _summary(g,total):
    n=len(g); wins=int(g.correct.sum()) if n else 0
    return {'picks':n,'coverage':float(n/max(total,1)),'hit_rate':float(wins/n) if n else 0.0,'wilson_lower95':_wilson_lower(wins,n),'avg_probability':float(g.pick_probability.mean()) if n else 0.0}

def _choose(train):
    dates=np.array(sorted(train.Date.unique()))
    if len(dates)<30:return CURRENT,[]
    cut=pd.Timestamp(dates[max(1,int(len(dates)*.70))-1]); val=train[train.Date>cut].copy()
    base=_summary(_candidate_rows(val,CURRENT),len(val)); min_picks=max(80,int(base['picks']*.70))
    trials=[]
    for cfg in GRID:
        g=_candidate_rows(val,cfg); s=_summary(g,len(val))
        if s['picks']<min_picks: continue
        # Precision-focused but penalize severe coverage loss; Wilson protects tiny samples.
        utility=s['wilson_lower95'] + .04*min(1.0,s['coverage']/max(base['coverage'],1e-6))
        trials.append({**cfg,**s,'utility':utility})
    if not trials:return CURRENT,[]
    trials.sort(key=lambda r:(r['utility'],r['hit_rate'],r['picks']),reverse=True)
    best={k:trials[0][k] for k in CURRENT}
    return best,trials[:25]

def main():
    d=pd.read_csv(SRC);d['Date']=pd.to_datetime(d.Date,errors='coerce').dt.normalize();d=d.dropna(subset=['Date','fold','actual_home_win','prob_ml','prob_mc']).sort_values('Date')
    rows=[];folds=[]
    for f,test in d.groupby('fold',sort=True):
        start=test.Date.min();train=d[d.Date<start].copy()
        if len(train)<500: continue
        cfg,trials=_choose(train)
        base=_candidate_rows(test,CURRENT);cand=_candidate_rows(test,cfg)
        # Emit one row per game with flags so gate can compare selective performance.
        z=test[['GameID','Date','fold','actual_home_win','prob_ml','prob_mc']].copy()
        bkeys=set(base.GameID.astype(int).tolist()); ckeys=set(cand.GameID.astype(int).tolist())
        bmap={int(r.GameID):(int(r.pick_home),int(r.correct),float(r.pick_probability)) for _,r in base.iterrows()}
        cmap={int(r.GameID):(int(r.pick_home),int(r.correct),float(r.pick_probability)) for _,r in cand.iterrows()}
        z['baseline_selected']=z.GameID.astype(int).map(lambda x:int(x in bkeys));z['candidate_selected']=z.GameID.astype(int).map(lambda x:int(x in ckeys))
        z['baseline_correct']=z.GameID.astype(int).map(lambda x:bmap.get(x,(0,np.nan,np.nan))[1]);z['candidate_correct']=z.GameID.astype(int).map(lambda x:cmap.get(x,(0,np.nan,np.nan))[1])
        z['baseline_pick_probability']=z.GameID.astype(int).map(lambda x:bmap.get(x,(0,np.nan,np.nan))[2]);z['candidate_pick_probability']=z.GameID.astype(int).map(lambda x:cmap.get(x,(0,np.nan,np.nan))[2])
        for k,v in cfg.items():z['selected_'+k]=v
        rows.append(z)
        folds.append({'fold':int(f),'train_rows':len(train),'test_rows':len(test),'selected':cfg,'baseline':_summary(base,len(test)),'candidate':_summary(cand,len(test)),'inner_trials':trials})
    if not rows:raise RuntimeError('No hubo folds validos para scanner')
    out=pd.concat(rows,ignore_index=True);OUT.parent.mkdir(parents=True,exist_ok=True);out.to_csv(OUT,index=False)
    rep={'policy':'nested_temporal_moneyline_scanner_thresholds_v1','historical_odds_available':False,'edge_ev_tested':False,'production_changed':False,'current':CURRENT,'rows':len(out),'folds':folds}
    REPORT.write_text(json.dumps(rep,indent=2),encoding='utf-8');print('SCANNER_THRESHOLD_WALKFORWARD',json.dumps(rep));print(f'OK {OUT} rows={len(out)}')
if __name__=='__main__':main()
