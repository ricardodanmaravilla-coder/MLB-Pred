"""Strict validation gate for confirmed-lineup experiments.

Uses paired DATE-BLOCK bootstrap rather than iid game bootstrap and requires
improvement in a majority of temporal folds.  This file is validation-only;
it does not alter production predictions.
"""
from __future__ import annotations
import argparse, json
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss


def _prob(s):
    x=pd.to_numeric(s,errors='coerce').astype(float)
    if x.dropna().max()>1.0: x=x/100.0
    return x.clip(1e-6,1-1e-6)

def _ece(y,p,bins=10):
    edges=np.linspace(0,1,bins+1); out=0.0
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(p>=lo)&((p<hi) if hi<1 else (p<=hi))
        if m.any(): out += m.mean()*abs(y[m].mean()-p[m].mean())
    return float(out)

def metrics(y,p):
    return {'brier':float(brier_score_loss(y,p)),'log_loss':float(log_loss(y,p,labels=[0,1])),'ece':_ece(y,p),'accuracy':float(((p>=.5).astype(int)==y).mean())}

def date_block_bootstrap(df,n=5000,seed=42):
    # Resample complete game dates, preserving within-day correlation.
    days=np.array(sorted(df['Date'].dropna().unique()))
    rng=np.random.default_rng(seed); gains=[]
    for _ in range(n):
        sampled=rng.choice(days,size=len(days),replace=True)
        chunks=[df[df.Date==d] for d in sampled]
        z=pd.concat(chunks,ignore_index=True)
        y=z.actual_home_win.to_numpy(int); b=z.baseline_home_prob.to_numpy(float); c=z.candidate_home_prob.to_numpy(float)
        gains.append(np.mean((b-y)**2)-np.mean((c-y)**2))
    a=np.asarray(gains)
    return {'mean_brier_gain':float(a.mean()),'p_candidate_better':float((a>0).mean()),'ci95':[float(x) for x in np.quantile(a,[.025,.975])]}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('csv'); ap.add_argument('--json-out'); args=ap.parse_args()
    d=pd.read_csv(args.csv); d['Date']=pd.to_datetime(d['Date'],errors='coerce').dt.date
    d['actual_home_win']=pd.to_numeric(d['actual_home_win'],errors='coerce')
    d['baseline_home_prob']=_prob(d['baseline_home_prob']); d['candidate_home_prob']=_prob(d['candidate_home_prob'])
    d=d.dropna(subset=['Date','actual_home_win','baseline_home_prob','candidate_home_prob']).copy(); d.actual_home_win=d.actual_home_win.astype(int)
    b=metrics(d.actual_home_win.to_numpy(),d.baseline_home_prob.to_numpy()); c=metrics(d.actual_home_win.to_numpy(),d.candidate_home_prob.to_numpy())
    folds=[]
    if 'fold' in d.columns:
        for f,g in d.groupby('fold'):
            bm=metrics(g.actual_home_win.to_numpy(),g.baseline_home_prob.to_numpy()); cm=metrics(g.actual_home_win.to_numpy(),g.candidate_home_prob.to_numpy())
            folds.append({'fold':int(f),'rows':len(g),'brier_gain':bm['brier']-cm['brier'],'logloss_gain':bm['log_loss']-cm['log_loss'],'ece_change':cm['ece']-bm['ece'],'improved_brier':cm['brier']<bm['brier'],'improved_logloss':cm['log_loss']<bm['log_loss']})
    boot=date_block_bootstrap(d)
    bg=b['brier']-c['brier']; lg=b['log_loss']-c['log_loss']; improved=sum(x['improved_brier'] and x['improved_logloss'] for x in folds)
    needed=(len(folds)//2+1) if folds else 1
    reasons=[]
    if len(d)<400: reasons.append('muestra_insuficiente')
    if bg<0.001: reasons.append('Brier_sin_mejora_suficiente')
    if lg<0.0005: reasons.append('LogLoss_sin_mejora_suficiente')
    if c['ece']>b['ece']+0.010: reasons.append('calibracion_empeora')
    if folds and improved<needed: reasons.append('sin_mejora_en_mayoria_de_folds')
    if boot['p_candidate_better']<0.90 or boot['ci95'][0]<=0: reasons.append('evidencia_dateblock_insuficiente')
    out={'rows':len(d),'dates':len(set(d.Date)),'baseline':b,'candidate':c,'gains':{'brier':bg,'log_loss':lg,'ece':b['ece']-c['ece']},'folds':folds,'folds_improved':improved,'folds_required':needed,'date_block_bootstrap':boot,'approved':not reasons,'decision':'PROMOTE' if not reasons else 'KEEP_BASELINE','reasons':reasons}
    print(json.dumps(out,indent=2));
    if args.json_out:
        from pathlib import Path
        Path(args.json_out).parent.mkdir(parents=True,exist_ok=True); Path(args.json_out).write_text(json.dumps(out,indent=2))
if __name__=='__main__': main()
