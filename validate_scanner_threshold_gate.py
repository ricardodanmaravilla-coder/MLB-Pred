"""Strict gate for validation-only moneyline scanner sports thresholds."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np,pandas as pd

def wilson_lower(w,n,z=1.96):
    if n<=0:return 0.0
    p=w/n;den=1+z*z/n;ctr=p+z*z/(2*n);adj=z*math.sqrt((p*(1-p)+z*z/(4*n))/n);return (ctr-adj)/den

def summarize(d,prefix):
    g=d[d[f'{prefix}_selected']==1].copy();n=len(g);wins=int(pd.to_numeric(g[f'{prefix}_correct'],errors='coerce').fillna(0).sum())
    return {'picks':n,'coverage':float(n/max(len(d),1)),'hit_rate':float(wins/n) if n else 0.0,'wilson_lower95':wilson_lower(wins,n)}

def date_bootstrap(d,draws=5000):
    daily=[]
    for _,g in d.groupby('Date',sort=True):
        b=g[g.baseline_selected==1];c=g[g.candidate_selected==1]
        if len(b)==0 or len(c)==0: continue
        bh=float(pd.to_numeric(b.baseline_correct,errors='coerce').mean());ch=float(pd.to_numeric(c.candidate_correct,errors='coerce').mean());daily.append(ch-bh)
    if len(daily)<20:return {'days':len(daily),'p_candidate_better':0.0,'ci95':[float('nan'),float('nan')],'mean_hit_gain':float('nan')}
    a=np.asarray(daily,float);r=np.random.default_rng(42);v=np.array([r.choice(a,len(a),replace=True).mean() for _ in range(draws)])
    return {'days':len(a),'p_candidate_better':float((v>0).mean()),'ci95':[float(np.quantile(v,.025)),float(np.quantile(v,.975))],'mean_hit_gain':float(a.mean())}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('csv');ap.add_argument('--json-out');a=ap.parse_args()
    d=pd.read_csv(a.csv);d['Date']=pd.to_datetime(d.Date,errors='coerce').dt.normalize();req=['Date','fold','baseline_selected','candidate_selected','baseline_correct','candidate_correct'];d=d.dropna(subset=['Date','fold'])
    bm=summarize(d,'baseline');cm=summarize(d,'candidate');gain=cm['hit_rate']-bm['hit_rate'];cov_ratio=cm['coverage']/max(bm['coverage'],1e-9)
    fs=[];good=0
    for f,g in d.groupby('fold',sort=True):
        b=summarize(g,'baseline');c=summarize(g,'candidate');ok=(c['hit_rate']>b['hit_rate']) and (c['coverage']>=.70*b['coverage']);good+=int(ok);fs.append({'fold':int(f),'baseline':b,'candidate':c,'hit_rate_gain':c['hit_rate']-b['hit_rate'],'coverage_ratio':c['coverage']/max(b['coverage'],1e-9),'improved':ok})
    bt=date_bootstrap(d);need=max(1,len(fs)//2+1);reasons=[]
    if bm['picks']<250 or cm['picks']<250:reasons.append('muestra_selectiva_insuficiente')
    if gain<0.015:reasons.append('precision_sin_mejora_suficiente')
    if cov_ratio<0.70:reasons.append('cobertura_demasiado_baja')
    if cm['wilson_lower95']<=bm['wilson_lower95']:reasons.append('wilson_no_mejora')
    if good<need:reasons.append('sin_mejora_en_mayoria_de_folds')
    if bt['p_candidate_better']<.90 or (np.isfinite(bt['ci95'][0]) and bt['ci95'][0]<=0):reasons.append('evidencia_dateblock_insuficiente')
    result={'rows':len(d),'baseline_current_sports_filter':bm,'candidate_nested_filter':cm,'hit_rate_gain':gain,'coverage_ratio':cov_ratio,'folds':fs,'folds_improved':good,'folds_required':need,'date_block_bootstrap':bt,'historical_odds_available':False,'edge_ev_tested':False,'approved':not reasons,'decision':'PROMOTE_SPORTS_THRESHOLDS' if not reasons else 'KEEP_CURRENT_SCANNER','reasons':reasons}
    text=json.dumps(result,indent=2);print(text)
    if a.json_out:Path(a.json_out).write_text(text,encoding='utf-8')
if __name__=='__main__':main()
