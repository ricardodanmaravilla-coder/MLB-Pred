"""Validation-only temporal calibration experiment for the production 50/50 MLB blend.

Fits calibrators only on dates strictly before each outer test fold.  Production is
never modified by this script.  Candidates: Platt/logistic and isotonic regression.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

SRC=Path('artifacts/ml_mc_blend_walkforward_predictions.csv')
OUT=Path('artifacts/probability_calibration_predictions.csv')
REPORT=Path('artifacts/probability_calibration_report.json')

def _clip(p): return np.clip(np.asarray(p,float),1e-6,1-1e-6)
def _logit(p):
    p=_clip(p); return np.log(p/(1-p)).reshape(-1,1)
def _score(y,p):
    p=_clip(p); y=np.asarray(y,int)
    return float(brier_score_loss(y,p)+0.15*log_loss(y,p,labels=[0,1]))
def _fit_predict(kind,ytr,ptr,pte):
    if kind=='platt':
        m=LogisticRegression(C=1.0,solver='lbfgs',max_iter=2000).fit(_logit(ptr),ytr)
        return m.predict_proba(_logit(pte))[:,1]
    m=IsotonicRegression(out_of_bounds='clip').fit(ptr,ytr)
    return m.predict(pte)

def main():
    df=pd.read_csv(SRC); df['Date']=pd.to_datetime(df['Date'],errors='coerce').dt.normalize()
    req=['Date','fold','actual_home_win','baseline_home_prob']
    miss=[c for c in req if c not in df.columns]
    if miss: raise RuntimeError(f'Faltan columnas: {miss}')
    df=df.dropna(subset=req).sort_values(['Date']).reset_index(drop=True)
    rows=[]; folds=[]
    for f,test in df.groupby('fold',sort=True):
        start=test.Date.min(); train=df[df.Date<start].copy()
        if len(train)<500: continue
        # Inner temporal tail chooses method without seeing outer test.
        dates=np.array(sorted(train.Date.unique()))
        cut=dates[max(1,int(len(dates)*0.75))-1]
        fit=train[train.Date<=cut]; val=train[train.Date>cut]
        if len(fit)<300 or len(val)<100: continue
        yfit=fit.actual_home_win.to_numpy(int); pfit=fit.baseline_home_prob.to_numpy(float)
        yval=val.actual_home_win.to_numpy(int); pval=val.baseline_home_prob.to_numpy(float)
        trials=[]
        for kind in ('platt','isotonic'):
            pred=_fit_predict(kind,yfit,pfit,pval)
            trials.append((kind,_score(yval,pred)))
        kind=min(trials,key=lambda x:x[1])[0]
        pred=_fit_predict(kind,train.actual_home_win.to_numpy(int),train.baseline_home_prob.to_numpy(float),test.baseline_home_prob.to_numpy(float))
        g=test.copy(); g['candidate_home_prob']=_clip(pred); g['calibration_method']=kind
        rows.append(g)
        folds.append({'fold':int(f),'train_rows':int(len(train)),'test_rows':int(len(test)),'method':kind,'inner_trials':[{'method':k,'score':s} for k,s in trials]})
    if not rows: raise RuntimeError('No hubo folds con historia suficiente para calibracion')
    out=pd.concat(rows,ignore_index=True)
    # validator expects these exact columns; baseline stays the production 50/50 probability.
    OUT.parent.mkdir(parents=True,exist_ok=True); out.to_csv(OUT,index=False)
    rep={'policy':'nested_temporal_probability_calibration_v1','production_changed':False,'rows':int(len(out)),'folds':folds,'methods':out.calibration_method.value_counts().to_dict()}
    REPORT.write_text(json.dumps(rep,indent=2),encoding='utf-8')
    print('CALIBRATION_WALKFORWARD',json.dumps(rep))
    print(f'OK {OUT} rows={len(out)}')
if __name__=='__main__': main()
