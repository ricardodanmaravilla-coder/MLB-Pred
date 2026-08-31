"""Strict promotion gate for the validation-only ML/Monte Carlo blend experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss


def _ece(y, p, bins=10):
    y=np.asarray(y,int); p=np.asarray(p,float); out=0.0
    edges=np.linspace(0,1,bins+1)
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(p>=lo)&((p<hi) if hi<1 else (p<=hi))
        if m.any(): out += float(m.mean())*abs(float(p[m].mean())-float(y[m].mean()))
    return float(out)


def _metrics(y,p):
    p=np.clip(np.asarray(p,float),1e-6,1-1e-6); y=np.asarray(y,int)
    return {
        "brier":float(brier_score_loss(y,p)),
        "log_loss":float(log_loss(y,p,labels=[0,1])),
        "ece":_ece(y,p),
        "accuracy":float(accuracy_score(y,(p>=0.5).astype(int))),
    }


def _date_block_bootstrap(df, draws=5000, seed=42):
    daily=[]
    for _,g in df.groupby("Date",sort=True):
        y=g["actual_home_win"].to_numpy(int)
        b=g["baseline_home_prob"].to_numpy(float)
        c=g["candidate_home_prob"].to_numpy(float)
        daily.append(float(np.mean((y-b)**2-(y-c)**2)))
    arr=np.asarray(daily,float)
    rng=np.random.default_rng(seed)
    vals=np.empty(draws,float)
    for i in range(draws): vals[i]=float(rng.choice(arr,size=len(arr),replace=True).mean())
    return {"mean_brier_gain":float(arr.mean()),"p_candidate_better":float(np.mean(vals>0)),"ci95":[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("--json-out",default=None); args=ap.parse_args()
    df=pd.read_csv(args.csv); df["Date"]=pd.to_datetime(df["Date"],errors="coerce").dt.normalize()
    req=["actual_home_win","baseline_home_prob","candidate_home_prob","fold","Date"]
    missing=[c for c in req if c not in df.columns]
    if missing: raise RuntimeError(f"Faltan columnas: {missing}")
    df=df.dropna(subset=req).copy()
    y=df.actual_home_win.to_numpy(int); b=df.baseline_home_prob.to_numpy(float); c=df.candidate_home_prob.to_numpy(float)
    bm=_metrics(y,b); cm=_metrics(y,c)
    gains={"brier":bm["brier"]-cm["brier"],"log_loss":bm["log_loss"]-cm["log_loss"],"ece":bm["ece"]-cm["ece"]}
    folds=[]; improved=0
    for f,g in df.groupby("fold",sort=True):
        yy=g.actual_home_win.to_numpy(int); bb=g.baseline_home_prob.to_numpy(float); cc=g.candidate_home_prob.to_numpy(float)
        mb=_metrics(yy,bb); mc=_metrics(yy,cc); bg=mb["brier"]-mc["brier"]; lg=mb["log_loss"]-mc["log_loss"]
        ok=bg>0 and lg>0; improved+=int(ok)
        folds.append({"fold":int(f),"rows":int(len(g)),"brier_gain":bg,"logloss_gain":lg,"improved_brier":bg>0,"improved_logloss":lg>0})
    required=max(1,len(folds)//2+1); boot=_date_block_bootstrap(df)
    selected=[]
    if "selected_ml_weight" in df.columns:
        selected=sorted({round(float(x),2) for x in pd.to_numeric(df.selected_ml_weight,errors="coerce").dropna()})
    reasons=[]
    if len(df)<1200: reasons.append("muestra_insuficiente")
    if gains["brier"]<0.0010: reasons.append("Brier_sin_mejora_suficiente")
    if gains["log_loss"]<0.0005: reasons.append("LogLoss_sin_mejora_suficiente")
    if cm["ece"]>bm["ece"]+0.010: reasons.append("calibracion_empeora")
    if improved<required: reasons.append("sin_mejora_en_mayoria_de_folds")
    if boot["p_candidate_better"]<0.90 or boot["ci95"][0]<=0: reasons.append("evidencia_dateblock_insuficiente")
    if selected and max(selected)-min(selected)>0.40: reasons.append("peso_inestable_entre_folds")
    result={"rows":int(len(df)),"dates":int(df.Date.nunique()),"baseline_50_50":bm,"candidate_nested_weight":cm,"gains":gains,"folds":folds,"folds_improved":improved,"folds_required":required,"selected_ml_weights":selected,"date_block_bootstrap":boot,"approved":not reasons,"decision":"PROMOTE" if not reasons else "KEEP_BASELINE","reasons":reasons}
    text=json.dumps(result,indent=2); print(text)
    if args.json_out: Path(args.json_out).write_text(text,encoding="utf-8")

if __name__=="__main__": main()
