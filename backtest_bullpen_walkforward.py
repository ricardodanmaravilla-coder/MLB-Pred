"""Walk-forward validation of real pregame bullpen availability.

The candidate never sees same-day or future reliever usage. For every game, bullpen
features are reconstructed only from completed MLB game feeds with Date < game Date.
Feature/model choice is nested inside each rolling fold; pooled out-of-fold predictions
are then evaluated by the existing promotion gate.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from modules.historical_mlb import prepare_games
from modules.metric_quality import batting_metric, pitching_metric
from modules.ml_mlb import PredictorMLMLB
from modules.team_utils import normalize_team

OUT = Path("artifacts/bullpen_walkforward_predictions.csv")
REPORT = Path("artifacts/bullpen_walkforward_report.json")
C_GRID = (0.01, 0.03, 0.10, 0.30)
FEATURE_SETS = {
    "usage_load": ("bp_pitches1_diff", "bp_pitches3_diff"),
    "arms_used": ("bp_arms1_diff", "bp_arms3_diff", "bp_b2b_diff"),
    "heavy_arms": ("bp_heavy2_diff", "bp_heavy3_diff"),
    "quality_at_risk": ("bp_quality_tired_diff",),
    "availability_core": ("bp_pitches1_diff", "bp_pitches3_diff", "bp_b2b_diff", "bp_heavy2_diff"),
    "availability_all": (
        "bp_pitches1_diff", "bp_pitches3_diff", "bp_arms1_diff", "bp_arms3_diff",
        "bp_b2b_diff", "bp_heavy2_diff", "bp_heavy3_diff", "bp_quality_tired_diff",
    ),
}


def _date_boundary(games, frac=.70):
    days = pd.Series(games["Date"].dt.normalize().dropna().unique()).sort_values().reset_index(drop=True)
    i = max(1, min(len(days)-1, int(len(days)*frac)))
    return pd.Timestamp(days.iloc[i])


def _maps(df, metric):
    x=df.copy(); x["Team"]=x["Team"].map(normalize_team); x["Season"]=pd.to_numeric(x["Season"],errors="coerce"); x[metric]=pd.to_numeric(x[metric],errors="coerce")
    return x.dropna(subset=["Team","Season",metric]).set_index(["Team","Season"])[metric].to_dict()


def _logit(p):
    p=np.clip(np.asarray(p,float),1e-5,1-1e-5); return np.log(p/(1-p))


def _design(df, cols):
    base=_logit(df["baseline_home_prob"].to_numpy(float)).reshape(-1,1)
    extra=df[list(cols)].to_numpy(float) if cols else np.empty((len(df),0))
    return np.hstack([base,extra])


def _brier(y,p): return float(np.mean((np.asarray(p,float)-np.asarray(y,float))**2))

def _logloss(y,p):
    p=np.clip(np.asarray(p,float),1e-9,1-1e-9); y=np.asarray(y,float)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))


def _ece(y,p,bins=10):
    y=np.asarray(y,float); p=np.asarray(p,float); edges=np.linspace(0,1,bins+1); out=0.0
    for i in range(bins):
        m=(p>=edges[i]) & ((p<edges[i+1]) if i<bins-1 else (p<=edges[i+1]))
        if np.any(m): out += (m.sum()/len(y))*abs(float(p[m].mean()-y[m].mean()))
    return float(out)


def _fit(df, cols, c):
    m=make_pipeline(StandardScaler(),LogisticRegression(C=float(c),max_iter=2000,solver="lbfgs",random_state=42))
    m.fit(_design(df,cols),df["actual_home_win"].to_numpy(int)); return m


def _prep_usage(path):
    u=pd.read_csv(path); u["Date"]=pd.to_datetime(u["Date"],errors="coerce"); u["Team"]=u["Team"].map(normalize_team)
    for c in ("Pitches","BF","ER","BB","SO","HR","IP"):
        if c in u.columns: u[c]=pd.to_numeric(u[c],errors="coerce")
    return u.dropna(subset=["Date","Team","PitcherID"]).sort_values(["Team","Date","GameID"])


def _quality_scores(prior):
    """Season-to-date reliever quality using only prior appearances; higher is better."""
    if prior.empty: return {}
    g=prior.groupby("PitcherID",as_index=False)[["BF","BB","SO","HR","ER","IP"]].sum(min_count=1)
    g=g[(g["BF"]>=25) & (g["IP"]>=8)]
    if g.empty: return {}
    kbb=(g["SO"].fillna(0)-g["BB"].fillna(0))/g["BF"].replace(0,np.nan)
    hr9=9*g["HR"].fillna(0)/g["IP"].replace(0,np.nan)
    era=9*g["ER"].fillna(0)/g["IP"].replace(0,np.nan)
    raw=1.8*kbb - .055*hr9 - .015*era
    vals=raw.replace([np.inf,-np.inf],np.nan).dropna()
    if len(vals)<3: return {}
    med=float(vals.median()); scale=float(vals.quantile(.75)-vals.quantile(.25)) or float(vals.std()) or 1.0
    return {int(pid):float(np.clip((v-med)/(2.5*scale),-.5,.5)) for pid,v in zip(g["PitcherID"],raw) if pd.notna(v)}


def _team_state(team_usage, date):
    prior=team_usage[team_usage["Date"]<date]
    if prior.empty:
        return {k:0.0 for k in ("pitches1","pitches3","arms1","arms3","b2b","heavy2","heavy3","quality_tired")}
    d=pd.Timestamp(date).normalize()
    r1=prior[prior["Date"]>=d-pd.Timedelta(days=1)]
    r2=prior[prior["Date"]>=d-pd.Timedelta(days=2)]
    r3=prior[prior["Date"]>=d-pd.Timedelta(days=3)]
    def ps(x): return float(pd.to_numeric(x.get("Pitches"),errors="coerce").fillna(0).sum()) if not x.empty else 0.0
    def arms(x): return float(x["PitcherID"].nunique()) if not x.empty else 0.0
    by2=r2.groupby("PitcherID")["Pitches"].sum(min_count=1) if not r2.empty else pd.Series(dtype=float)
    by3=r3.groupby("PitcherID")["Pitches"].sum(min_count=1) if not r3.empty else pd.Series(dtype=float)
    dates_by=prior[prior["Date"]>=d-pd.Timedelta(days=2)].groupby("PitcherID")["Date"].nunique()
    b2b=float((dates_by>=2).sum()) if len(dates_by) else 0.0
    heavy2=float((by2.fillna(0)>=30).sum()) if len(by2) else 0.0
    heavy3=float((by3.fillna(0)>=45).sum()) if len(by3) else 0.0
    q=_quality_scores(prior)
    tired=0.0
    for pid,pitches in by2.fillna(0).items():
        if pitches>=20:
            tired += max(0.0,q.get(int(pid),0.0)) * min(float(pitches)/40.0,1.5)
    return {
        "pitches1":ps(r1)/100.0, "pitches3":ps(r3)/220.0,
        "arms1":arms(r1)/5.0, "arms3":arms(r3)/10.0,
        "b2b":b2b/3.0, "heavy2":heavy2/3.0, "heavy3":heavy3/4.0,
        "quality_tired":float(tired),
    }


def _build_states(usage, games):
    teams={t:g.copy() for t,g in usage.groupby("Team")}
    cache={}
    out=[]
    for _,r in games.iterrows():
        d=pd.Timestamp(r["Date"]).normalize(); h=normalize_team(r["Home"]); a=normalize_team(r["Away"])
        def state(t):
            key=(t,d)
            if key not in cache: cache[key]=_team_state(teams.get(t,pd.DataFrame(columns=usage.columns)),d)
            return cache[key]
        hs,as_=state(h),state(a)
        # Positive diff means HOME bullpen is more fatigued / more quality is at risk.
        feats={f"bp_{k}_diff":hs[k]-as_[k] for k in hs}
        feats["bullpen_usage_coverage"]=float((h in teams) and (a in teams))
        out.append(feats)
    return pd.DataFrame(out,index=games.index)


def _select(train):
    days=pd.Series(train["Date"].dt.normalize().unique()).sort_values().reset_index(drop=True)
    cut=pd.Timestamp(days.iloc[max(1,min(len(days)-1,int(len(days)*.75)))])
    tr=train[train["Date"].dt.normalize()<cut]; va=train[train["Date"].dt.normalize()>=cut]
    if len(tr)<500 or len(va)<200: raise RuntimeError(f"Inner temporal split insuficiente {len(tr)}/{len(va)}")
    y=va["actual_home_win"].to_numpy(int); base=va["baseline_home_prob"].to_numpy(float)
    base_b=_brier(y,base); base_l=_logloss(y,base); base_e=_ece(y,base)
    trials=[]
    for name,cols in FEATURE_SETS.items():
        for c in C_GRID:
            m=_fit(tr,cols,c); p=m.predict_proba(_design(va,cols))[:,1]
            b,l,e=_brier(y,p),_logloss(y,p),_ece(y,p)
            score=b+.15*l+.10*max(0,e-base_e)+.20*max(0,b-base_b)
            trials.append({"name":name,"C":c,"features":list(cols),"score":score,"brier":b,"log_loss":l,"ece":e})
    trials.sort(key=lambda z:(z["score"],z["brier"],z["log_loss"],len(z["features"])))
    return trials[0], trials[:5], str(cut.date())


def main():
    usage_path=Path("data/mlb_bullpen_usage_history.csv")
    if not usage_path.exists(): raise RuntimeError("Ejecuta primero build_bullpen_usage_history.py")
    bat=pd.read_csv("data/mlb_batting.csv"); pit=pd.read_csv("data/mlb_pitching.csv"); games=prepare_games(pd.read_csv("data/mlb_games.csv"))
    d1=_date_boundary(games,.70); train=games[games["Date"].dt.normalize()<d1].copy(); later=games[games["Date"].dt.normalize()>=d1].copy()
    model=PredictorMLMLB()
    if not model.entrenar(bat,pit,train): raise RuntimeError("No se pudo entrenar baseline")
    bc,pc=batting_metric(bat),pitching_metric(pit)
    if not bc or not pc: raise RuntimeError("Métricas baseline inválidas")
    bd,pdct=_maps(bat,bc),_maps(pit,pc); bmed=float(pd.to_numeric(bat[bc],errors="coerce").median()); pmed=float(pd.to_numeric(pit[pc],errors="coerce").median())
    rows=[]
    for day,dg in later.groupby(later["Date"].dt.normalize(),sort=True):
        pending=[]
        for idx,r in dg.iterrows():
            h,a=normalize_team(r["Home"]),normalize_team(r["Away"]); year=int(r["Season"]); sy=year-1
            pred=model.predecir_partido(h,a,float(bd.get((h,sy),bmed)),float(bd.get((a,sy),bmed)),float(pdct.get((h,sy),pmed)),float(pdct.get((a,sy),pmed)),game_date=r["Date"])
            hs,aw=float(r["Home_Score"]),float(r["Away_Score"])
            rows.append({"idx":idx,"Date":r["Date"],"Home":h,"Away":a,"actual_home_win":int(hs>aw),"actual_total_runs":hs+aw,"baseline_home_prob":float(pred["Probabilidad_Local"])/100.0,"baseline_total_runs":float(pred["Proyeccion_Carreras"])})
            pending.append((h,a,hs,aw,r["Date"]))
        for h,a,hs,aw,gd in pending: model.actualizar_resultado(h,a,hs,aw,game_date=gd)
    frame=pd.DataFrame(rows).set_index("idx"); usage=_prep_usage(usage_path); feats=_build_states(usage,later.loc[frame.index])
    frame=frame.join(feats).sort_values("Date").reset_index(drop=True); frame["Date"]=pd.to_datetime(frame["Date"])
    days=pd.Series(frame["Date"].dt.normalize().unique()).sort_values().reset_index(drop=True)
    warm_i=max(1,int(len(days)*.45)); remaining=days.iloc[warm_i:].reset_index(drop=True)
    chunks=np.array_split(remaining,3); pooled=[]; fold_reports=[]
    for fi,chunk in enumerate(chunks,1):
        if len(chunk)==0: continue
        start=pd.Timestamp(chunk.iloc[0]); end=pd.Timestamp(chunk.iloc[-1])
        fit=frame[frame["Date"].dt.normalize()<start].copy(); test=frame[(frame["Date"].dt.normalize()>=start)&(frame["Date"].dt.normalize()<=end)].copy()
        if len(fit)<1000 or len(test)<300: continue
        best,top5,inner_cut=_select(fit); m=_fit(fit,best["features"],best["C"]); p=m.predict_proba(_design(test,best["features"]))[:,1]
        test["candidate_home_prob"]=p; test["candidate_total_runs"]=test["baseline_total_runs"]; test["walkforward_fold"]=fi; pooled.append(test)
        fold_reports.append({"fold":fi,"test_start":str(start.date()),"test_end":str(end.date()),"train_rows":len(fit),"test_rows":len(test),"inner_cut":inner_cut,"selected":best,"top5":top5})
    if not pooled: raise RuntimeError("No se pudieron generar folds walk-forward")
    out=pd.concat(pooled,ignore_index=True).sort_values("Date"); OUT.parent.mkdir(parents=True,exist_ok=True); out.to_csv(OUT,index=False)
    report={"policy":"nested_three_fold_walkforward_real_bullpen_usage","rows":len(out),"folds":fold_reports,"usage_rows":len(usage),"usage_games":int(usage["GameID"].nunique()),"coverage":float(out["bullpen_usage_coverage"].mean())}
    REPORT.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print("BULLPEN_WALKFORWARD",json.dumps(report,ensure_ascii=False))
    print(f"OK {OUT} rows={len(out)} coverage={report['coverage']:.1%}")


if __name__=="__main__": main()
