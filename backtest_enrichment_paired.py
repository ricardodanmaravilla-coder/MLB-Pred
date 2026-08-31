"""Leak-safe paired test of starter/platoon enrichment versus the current MLB baseline.

Design:
- earliest 70% of complete dates: train current PredictorMLMLB
- next 15%: train a small logistic meta-model using baseline probability + only
  pregame enrichment features from prior completed seasons
- final 15%: untouched holdout used for the paired decision gate

The historical starter file is queried with Season <= game_year-1. Pitcher hand may be
filled from the current roster file because throwing hand is a stable biographical trait,
not a future performance statistic. Missing data are neutral and accompanied by coverage.
"""
from __future__ import annotations

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

OUT = Path("artifacts/enrichment_paired_predictions.csv")

LOWER = {"FIP": 0.20, "xFIP": 0.30, "SIERA": 0.20, "WHIP": 0.12, "BB%": 0.05, "HR/9": 0.05}
HIGHER = {"K-BB%": 0.05, "K%": 0.03}


def _name_key(v):
    import unicodedata
    s = unicodedata.normalize("NFKD", str(v or ""))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return " ".join(s.casefold().replace(".", " ").replace("-", " ").split())


def _date_boundaries(games, a=0.70, b=0.85):
    days = pd.Series(games["Date"].dt.normalize().dropna().unique()).sort_values().reset_index(drop=True)
    if len(days) < 30:
        raise RuntimeError("Histórico insuficiente para split temporal de tres bloques")
    i = max(1, min(len(days)-2, int(len(days)*a)))
    j = max(i+1, min(len(days)-1, int(len(days)*b)))
    return pd.Timestamp(days.iloc[i]), pd.Timestamp(days.iloc[j])


def _maps(df, metric):
    x = df.copy(); x["Team"] = x["Team"].map(normalize_team)
    x["Season"] = pd.to_numeric(x["Season"], errors="coerce")
    x[metric] = pd.to_numeric(x[metric], errors="coerce")
    return x.dropna(subset=["Team", "Season", metric]).set_index(["Team", "Season"])[metric].to_dict()


def _starter_prior(history, name, team, game_year):
    if history.empty or not name or str(name) == "nan":
        return None
    x = history.copy()
    key = _name_key(name)
    m = x[x["_name_key"] == key]
    if team and not m.empty:
        mt = m[m["Team"].map(normalize_team) == normalize_team(team)]
        if not mt.empty: m = mt
    if m.empty:
        surname = key.split()[-1] if key else ""
        m = x[x["_name_key"].str.split().str[-1] == surname]
        if team and not m.empty:
            mt = m[m["Team"].map(normalize_team) == normalize_team(team)]
            if not mt.empty: m = mt
        if m["Name"].nunique() != 1:
            return None
    m = m[pd.to_numeric(m["Season"], errors="coerce") <= int(game_year)-1]
    if m.empty:
        return None
    return m.assign(_s=pd.to_numeric(m["Season"], errors="coerce")).sort_values("_s").iloc[-1]


def _starter_factor(row, population):
    if row is None:
        return 1.0, 0.0
    year = int(pd.to_numeric(pd.Series([row.get("Season")]), errors="coerce").iloc[0])
    pop = population[pd.to_numeric(population["Season"], errors="coerce") == year]
    total_w = sum(LOWER.values()) + sum(HIGHER.values())
    logsum = 0.0; used = 0.0
    for col, w in {**LOWER, **HIGHER}.items():
        if col not in pop.columns: continue
        v = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
        s = pd.to_numeric(pop[col], errors="coerce").dropna()
        if pd.isna(v) or len(s) < 30: continue
        c = float(s.median())
        if abs(c) < 1e-9: continue
        f = float(v)/c if col in LOWER else (c/float(v) if abs(float(v)) > 1e-9 else 1.0)
        f = float(np.clip(f, 0.82, 1.18))
        logsum += w * math.log(f); used += w
    return (float(math.exp(logsum/used)) if used else 1.0, float(used/total_w) if total_w else 0.0)


def _hand_map(current_pitchers):
    if current_pitchers.empty or "Name" not in current_pitchers.columns or "PitchHand" not in current_pitchers.columns:
        return {}
    out = {}
    for r in current_pitchers.itertuples(index=False):
        h = str(getattr(r, "PitchHand", "")).strip().upper()[:1]
        if h in {"L", "R"}: out[_name_key(getattr(r, "Name", ""))] = h
    return out


def _platoon_factor(batting, team, season, hand):
    if hand not in {"L", "R"}: return 1.0, 0.0
    x = batting.copy(); x["_t"] = x["Team"].map(normalize_team); x["_s"] = pd.to_numeric(x["Season"], errors="coerce")
    sy = int(season)-1; rows = x[(x["_t"] == normalize_team(team)) & (x["_s"] == sy)]
    season_rows = x[x["_s"] == sy]
    if rows.empty or season_rows.empty: return 1.0, 0.0
    row = rows.iloc[-1]; specs=[(f"OPS_vs_{hand}",.50),(f"OBP_vs_{hand}",.25),(f"SLG_vs_{hand}",.25)]
    logsum=0.0; used=0.0
    for col,w in specs:
        if col not in x.columns: continue
        v=pd.to_numeric(pd.Series([row.get(col)]),errors="coerce").iloc[0]; s=pd.to_numeric(season_rows[col],errors="coerce").dropna()
        if pd.isna(v) or len(s)<20: continue
        c=float(s.median())
        if c<=0: continue
        logsum += w*math.log(float(np.clip(float(v)/c,.80,1.20))); used += w
    return (float(math.exp(logsum/used)) if used else 1.0, min(1.0,used))


def _extra_features(row, batting, pitcher_hist, hands):
    year=int(row["Season"]); h=normalize_team(row["Home"]); a=normalize_team(row["Away"])
    hs=str(row.get("Home_Starter", "")); as_=str(row.get("Away_Starter", ""))
    hr=_starter_prior(pitcher_hist,hs,h,year); ar=_starter_prior(pitcher_hist,as_,a,year)
    hf,hcov=_starter_factor(hr,pitcher_hist); af,acov=_starter_factor(ar,pitcher_hist)
    hh=hands.get(_name_key(hs)); ah=hands.get(_name_key(as_))
    home_bat,hbcov=_platoon_factor(batting,h,year,ah); away_bat,abcov=_platoon_factor(batting,a,year,hh)
    return np.asarray([
        math.log(max(.80,min(1.20,home_bat))), math.log(max(.80,min(1.20,away_bat))),
        math.log(max(.82,min(1.18,hf))), math.log(max(.82,min(1.18,af))),
        hbcov, abcov, hcov, acov,
    ],dtype=float)


def main():
    bat=pd.read_csv("data/mlb_batting.csv"); pit=pd.read_csv("data/mlb_pitching.csv")
    games=prepare_games(pd.read_csv("data/mlb_games.csv"))
    ph_path=Path("data/mlb_pitching_individual_history.csv")
    if not ph_path.exists(): raise RuntimeError("Ejecuta primero build_pitcher_history.py")
    ph=pd.read_csv(ph_path); ph["_name_key"]=ph["Name"].map(_name_key)
    current_path=Path("data/mlb_pitching_individual.csv")
    current=pd.read_csv(current_path) if current_path.exists() else pd.DataFrame(); hands=_hand_map(current)
    d1,d2=_date_boundaries(games); train=games[games["Date"].dt.normalize()<d1].copy(); later=games[games["Date"].dt.normalize()>=d1].copy()
    model=PredictorMLMLB()
    if not model.entrenar(bat,pit,train): raise RuntimeError("No se pudo entrenar baseline")
    bc=batting_metric(bat); pc=pitching_metric(pit)
    if not bc or not pc: raise RuntimeError("Métricas base no válidas")
    bd=_maps(bat,bc); pdict=_maps(pit,pc); bmed=float(pd.to_numeric(bat[bc],errors="coerce").median()); pmed=float(pd.to_numeric(pit[pc],errors="coerce").median())
    rows=[]
    for day,day_games in later.groupby(later["Date"].dt.normalize(),sort=True):
        pending=[]
        for _,r in day_games.iterrows():
            h=normalize_team(r["Home"]); a=normalize_team(r["Away"]); year=int(r["Season"]); sy=year-1
            pred=model.predecir_partido(h,a,float(bd.get((h,sy),bmed)),float(bd.get((a,sy),bmed)),float(pdict.get((h,sy),pmed)),float(pdict.get((a,sy),pmed)),game_date=r["Date"])
            pb=float(pred["Probabilidad_Local"])/100.0; extra=_extra_features(r,bat,ph,hands)
            hs=float(r["Home_Score"]); aw=float(r["Away_Score"])
            rows.append({"Date":r["Date"],"actual_home_win":int(hs>aw),"actual_total_runs":hs+aw,"baseline_home_prob":pb,"baseline_total_runs":float(pred["Proyeccion_Carreras"]),"_extra":extra})
            pending.append((h,a,hs,aw,r["Date"]))
        for h,a,hs,aw,gd in pending: model.actualizar_resultado(h,a,hs,aw,game_date=gd)
    frame=pd.DataFrame(rows); frame["Date"]=pd.to_datetime(frame["Date"])
    fit=frame[frame["Date"].dt.normalize()<d2].copy(); test=frame[frame["Date"].dt.normalize()>=d2].copy()
    if len(fit)<300 or len(test)<400: raise RuntimeError(f"Muestra insuficiente fit={len(fit)} test={len(test)}")
    def design(df):
        p=np.clip(df["baseline_home_prob"].to_numpy(float),1e-5,1-1e-5); logit=np.log(p/(1-p)).reshape(-1,1)
        ex=np.vstack(df["_extra"].to_numpy()); return np.hstack([logit,ex])
    meta=make_pipeline(StandardScaler(),LogisticRegression(C=.20,max_iter=2000,solver="lbfgs",random_state=42))
    meta.fit(design(fit),fit["actual_home_win"].to_numpy(int))
    pcand=meta.predict_proba(design(test))[:,1]
    out=test.drop(columns=["_extra"]).copy(); out["candidate_home_prob"]=pcand
    # Total candidate is deliberately left equal to baseline until total-specific
    # enrichment is independently validated; this prevents a moneyline experiment
    # from being promoted on an untested totals transformation.
    out["candidate_total_runs"]=out["baseline_total_runs"]
    OUT.parent.mkdir(parents=True,exist_ok=True); out.to_csv(OUT,index=False)
    print(f"OK paired holdout: {OUT} rows={len(out)} fit={len(fit)} holdout_start={d2.date()}")


if __name__=="__main__": main()
