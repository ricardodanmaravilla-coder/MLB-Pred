"""Strict date-block promotion gate for probability calibration."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import brier_score_loss,log_loss,accuracy_score

def ece(y,p,bins=10):
    y=np.asarray(y,int);p=np.asarray(p,float);out=0.; edges=np.linspace(0,1,bins+1)
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(p>=lo)&((p<hi) if hi<1 else (p<=hi))
        if m.any(): out+=float(m.mean())*abs(float(p[m].mean())-float(y[m].mean()))
    return float(out)
def metrics(y,p):
    p=np.clip(np.asarray(p,float),1e-6,1-1e-6);y=np.asarray(y,int)
    return {'brier':float(brier_score_loss(y,p)),'log_loss':float(log_loss(y,p,labels=[0,1])),'ece':ece(y,p),'accuracy':float(accuracy_score(y,(p>=.5).astype(int)))}
def boot(df,n=5000):
    daily=[]
    for _,g in df.groupby('Date'):
        y=g.actual_home_win.to_numpy(int);b=g.baseline_home_prob.to_numpy(float);c=g.candidate_home_prob.to_numpy(float)
        daily.append(float(np.mean((y-b)**2-(y-c)**2)))
    a=np.asarray(daily);r=np.random.default_rng(42);v=np.array([r.choice(a,len(a),replace=True).mean() for _ in range(n)])
    return {'p_candidate_better':float((v>0).mean()),'ci95':[float(np.quantile(v,.025)),float(np.quantile(v,.975))],'mean_brier_gain':float(a.mean())}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('csv');ap.add_argument('--json-out');a=ap.parse_args()
    d=pd.read_csv(a.csv);d['Date']=pd.to_datetime(d.Date,errors='coerce').dt.normalize();d=d.dropna(subset=['Date','fold','actual_home_win','baseline_home_prob','candidate_home_prob'])
    bm=metrics(d.actual_home_win,d.baseline_home_prob);cm=metrics(d.actual_home_win,d.candidate_home_prob)
    gain={'brier':bm['brier']-cm['brier'],'log_loss':bm['log_loss']-cm['log_loss'],'ece':bm['ece']-cm['ece']}
    fs=[];good=0
    for f,g in d.groupby('fold'):
        x=metrics(g.actual_home_win,g.baseline_home_prob);z=metrics(g.actual_home_win,g.candidate_home_prob);ok=x['brier']>z['brier'] and x['log_loss']>z['log_loss'];good+=int(ok);fs.append({'fold':int(f),'rows':len(g),'brier_gain':x['brier']-z['brier'],'logloss_gain':x['log_loss']-z['log_loss'],'ece_gain':x['ece']-z['ece'],'improved':ok})
    bt=boot(d);need=max(1,len(fs)//2+1);reasons=[]
    if len(d)<1200:reasons.append('muestra_insuficiente')
    if gain['brier']<0.0005:reasons.append('Brier_sin_mejora_suficiente')
    if gain['log_loss']<0.0005:reasons.append('LogLoss_sin_mejora_suficiente')
    if gain['ece']<=0:reasons.append('ECE_no_mejora')
    if good<need:reasons.append('sin_mejora_en_mayoria_de_folds')
    if bt['p_candidate_better']<.90 or bt['ci95'][0]<=0:reasons.append('evidencia_dateblock_insuficiente')
    r={'rows':len(d),'dates':d.Date.nunique(),'baseline':bm,'candidate_calibrated':cm,'gains':gain,'folds':fs,'folds_improved':good,'folds_required':need,'date_block_bootstrap':bt,'approved':not reasons,'decision':'PROMOTE' if not reasons else 'KEEP_BASELINE','reasons':reasons}
    text=json.dumps(r,indent=2);print(text)
    if a.json_out:Path(a.json_out).write_text(text,encoding='utf-8')
if __name__=='__main__':main()
