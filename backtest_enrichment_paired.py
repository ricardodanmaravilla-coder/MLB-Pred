"""Leak-safe feature search for MLB starter/platoon enrichment.

The production baseline is never replaced merely because more variables exist.
This script uses three temporal layers:
1) earliest 70% of dates: train the existing MLB baseline;
2) next 15%: generate pregame enrichment observations and perform an INNER
   chronological model/feature search without touching the final holdout;
3) final 15%: one untouched evaluation used by validate_enrichment.py.

Pitcher identity is joined by official MLB GameID -> PlayerID. Pitcher performance
is restricted to seasons <= game_year-1. Missing data stay neutral. The search uses
compact matchup DIFFERENTIALS instead of separate home/away values and never feeds
coverage flags as predictive variables; this reduces dimensionality and avoids
learning accidental missing-data patterns.
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

OUT = Path("artifacts/enrichment_paired_predictions.csv")
REPORT = Path("artifacts/enrichment_feature_search.json")

LOWER_BETTER = ("FIP", "ERA", "WHIP", "BB%", "HR/9")
HIGHER_BETTER = ("K-BB%", "K%", "GB%")
PITCH_METRICS = LOWER_BETTER + HIGHER_BETTER
PLATOON_METRICS = ("OPS", "OBP", "SLG")

# Candidate families are deliberately small. Every set includes baseline logit.
FEATURE_SETS = {
    "platoon_ops": ("platoon_OPS_diff",),
    "platoon_all": tuple(f"platoon_{m}_diff" for m in PLATOON_METRICS),
    "starter_fip": ("pitch_FIP_diff",),
    "starter_whip": ("pitch_WHIP_diff",),
    "starter_kbb": ("pitch_K-BB%_diff",),
    "starter_core": ("pitch_FIP_diff", "pitch_WHIP_diff", "pitch_K-BB%_diff"),
    "starter_contact": ("pitch_HR/9_diff", "pitch_GB%_diff"),
    "starter_all": tuple(f"pitch_{m}_diff" for m in PITCH_METRICS),
    "core_plus_platoon": (
        "pitch_FIP_diff", "pitch_WHIP_diff", "pitch_K-BB%_diff", "platoon_OPS_diff"
    ),
    "all_compact": tuple(f"pitch_{m}_diff" for m in PITCH_METRICS)
        + tuple(f"platoon_{m}_diff" for m in PLATOON_METRICS),
}
C_GRID = (0.03, 0.10, 0.30, 1.00)


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


def _starter_prior(history, player_id, name, team, game_year):
    if history.empty:
        return None
    m = pd.DataFrame()
    pid = pd.to_numeric(pd.Series([player_id]), errors="coerce").iloc[0]
    if pd.notna(pid) and "PlayerID" in history.columns:
        ids = pd.to_numeric(history["PlayerID"], errors="coerce")
        m = history[ids == int(pid)]
    if m.empty and name and str(name) != "nan":
        key = _name_key(name)
        m = history[history["_name_key"] == key]
        if team and not m.empty:
            mt = m[m["Team"].map(normalize_team) == normalize_team(team)]
            if not mt.empty:
                m = mt
        if m.empty:
            surname = key.split()[-1] if key else ""
            m = history[history["_name_key"].str.split().str[-1] == surname]
            if team and not m.empty:
                mt = m[m["Team"].map(normalize_team) == normalize_team(team)]
                if not mt.empty:
                    m = mt
            if m.empty or m["Name"].nunique() != 1:
                return None
    if m.empty:
        return None
    m = m[pd.to_numeric(m["Season"], errors="coerce") <= int(game_year)-1]
    if m.empty:
        return None
    return m.assign(_s=pd.to_numeric(m["Season"], errors="coerce")).sort_values("_s").iloc[-1]


def _reference_population(population, season):
    pop = population[pd.to_numeric(population["Season"], errors="coerce") == int(season)].copy()
    if "IP" in pop.columns:
        ip = pd.to_numeric(pop["IP"], errors="coerce")
        eligible = pop[ip >= 20.0]
        if len(eligible) >= 30:
            pop = eligible
    return pop


def _pitch_signal(row, population, metric):
    """Positive means better than that season's pitcher median."""
    if row is None or metric not in population.columns:
        return 0.0, 0.0
    season = pd.to_numeric(pd.Series([row.get("Season")]), errors="coerce").iloc[0]
    value = pd.to_numeric(pd.Series([row.get(metric)]), errors="coerce").iloc[0]
    if pd.isna(season) or pd.isna(value):
        return 0.0, 0.0
    pop = _reference_population(population, int(season))
    vals = pd.to_numeric(pop[metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(vals) < 30:
        return 0.0, 0.0
    median = float(vals.median())
    v = float(value)
    if metric in LOWER_BETTER:
        if v <= 0 or median <= 0:
            return 0.0, 0.0
        ratio = median / v
    else:
        # K-BB% can be near/under zero for poor pitchers; use robust z-like scaling.
        scale = float(vals.quantile(.75) - vals.quantile(.25))
        if not math.isfinite(scale) or scale < 1e-6:
            scale = float(vals.std())
        if not math.isfinite(scale) or scale < 1e-6:
            return 0.0, 0.0
        return float(np.clip((v - median) / (3.0 * scale), -.25, .25)), 1.0
    return float(np.clip(math.log(np.clip(ratio, .75, 1.25)), -.25, .25)), 1.0


def _row_hand(row):
    if row is None:
        return None
    h = str(row.get("PitchHand", "") or "").strip().upper()[:1]
    return h if h in {"L", "R"} else None


def _platoon_signals(batting, team, season, hand):
    out = {m: 0.0 for m in PLATOON_METRICS}; coverage = {m: 0.0 for m in PLATOON_METRICS}
    if hand not in {"L", "R"}:
        return out, coverage
    x = batting.copy(); x["_t"] = x["Team"].map(normalize_team); x["_s"] = pd.to_numeric(x["Season"], errors="coerce")
    sy = int(season)-1; rows = x[(x["_t"] == normalize_team(team)) & (x["_s"] == sy)]; season_rows = x[x["_s"] == sy]
    if rows.empty or season_rows.empty:
        return out, coverage
    row = rows.iloc[-1]
    for metric in PLATOON_METRICS:
        col = f"{metric}_vs_{hand}"
        if col not in x.columns:
            continue
        value = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
        vals = pd.to_numeric(season_rows[col], errors="coerce").replace([np.inf,-np.inf],np.nan).dropna()
        if pd.isna(value) or len(vals) < 20:
            continue
        median = float(vals.median())
        if median <= 0 or float(value) <= 0:
            continue
        out[metric] = float(np.clip(math.log(np.clip(float(value)/median, .80, 1.20)), -.23, .23))
        coverage[metric] = 1.0
    return out, coverage


def _extra_features(row, batting, pitcher_hist):
    year=int(row["Season"]); h=normalize_team(row["Home"]); a=normalize_team(row["Away"])
    hs=str(row.get("Home_Starter", "")); as_=str(row.get("Away_Starter", ""))
    hid=row.get("HomeStarterID"); aid=row.get("AwayStarterID")
    hr=_starter_prior(pitcher_hist,hid,hs,h,year); ar=_starter_prior(pitcher_hist,aid,as_,a,year)
    hh=_row_hand(hr); ah=_row_hand(ar)

    features = {}
    pitch_cov = []
    for metric in PITCH_METRICS:
        home_sig, hc = _pitch_signal(hr, pitcher_hist, metric)
        away_sig, ac = _pitch_signal(ar, pitcher_hist, metric)
        # Positive differential favours the home team.
        features[f"pitch_{metric}_diff"] = home_sig - away_sig
        pitch_cov.extend([hc, ac])

    home_bat, home_cov = _platoon_signals(batting,h,year,ah)
    away_bat, away_cov = _platoon_signals(batting,a,year,hh)
    for metric in PLATOON_METRICS:
        features[f"platoon_{metric}_diff"] = home_bat[metric] - away_bat[metric]

    diagnostics = {
        "home_pitcher_coverage": float(np.mean([_pitch_signal(hr,pitcher_hist,m)[1] for m in PITCH_METRICS])),
        "away_pitcher_coverage": float(np.mean([_pitch_signal(ar,pitcher_hist,m)[1] for m in PITCH_METRICS])),
        "home_platoon_coverage": float(np.mean(list(home_cov.values()))),
        "away_platoon_coverage": float(np.mean(list(away_cov.values()))),
        "home_exact_id": float(pd.notna(pd.to_numeric(pd.Series([hid]), errors="coerce").iloc[0])),
        "away_exact_id": float(pd.notna(pd.to_numeric(pd.Series([aid]), errors="coerce").iloc[0])),
    }
    return features, diagnostics


def _logit(p):
    p=np.clip(np.asarray(p,dtype=float),1e-5,1-1e-5)
    return np.log(p/(1-p))


def _design(df, cols):
    base=_logit(df["baseline_home_prob"].to_numpy(float)).reshape(-1,1)
    extras=df[list(cols)].to_numpy(float) if cols else np.empty((len(df),0))
    return np.hstack([base, extras])


def _brier(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float)
    return float(np.mean((p-y)**2))


def _logloss(y,p):
    y=np.asarray(y,float); p=np.clip(np.asarray(p,float),1e-9,1-1e-9)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))


def _ece(y,p,bins=10):
    y=np.asarray(y,float); p=np.asarray(p,float); edges=np.linspace(0,1,bins+1); total=len(y); out=0.0
    for i in range(bins):
        mask=(p>=edges[i]) & ((p<edges[i+1]) if i<bins-1 else (p<=edges[i+1]))
        if np.any(mask): out += (mask.sum()/total)*abs(float(p[mask].mean()-y[mask].mean()))
    return float(out)


def _fit_meta(train, cols, c):
    model=make_pipeline(StandardScaler(),LogisticRegression(C=float(c),max_iter=2000,solver="lbfgs",random_state=42))
    model.fit(_design(train,cols),train["actual_home_win"].to_numpy(int))
    return model


def _select_features(fit):
    """Select only on an inner chronological validation slice."""
    days=pd.Series(fit["Date"].dt.normalize().unique()).sort_values().reset_index(drop=True)
    cut=pd.Timestamp(days.iloc[max(1,min(len(days)-1,int(len(days)*.60)))])
    inner_train=fit[fit["Date"].dt.normalize()<cut].copy(); inner_val=fit[fit["Date"].dt.normalize()>=cut].copy()
    if len(inner_train)<500 or len(inner_val)<250:
        raise RuntimeError(f"Inner split insuficiente train={len(inner_train)} val={len(inner_val)}")
    y=inner_val["actual_home_win"].to_numpy(int)
    baseline=inner_val["baseline_home_prob"].to_numpy(float)
    baseline_metrics={"brier":_brier(y,baseline),"log_loss":_logloss(y,baseline),"ece":_ece(y,baseline)}
    trials=[]
    for name,cols in FEATURE_SETS.items():
        for c in C_GRID:
            model=_fit_meta(inner_train,cols,c); p=model.predict_proba(_design(inner_val,cols))[:,1]
            metrics={"brier":_brier(y,p),"log_loss":_logloss(y,p),"ece":_ece(y,p)}
            # Primary predictive loss, with a small calibration penalty. No accuracy tuning.
            score=metrics["brier"] + .15*metrics["log_loss"] + .10*max(0.0,metrics["ece"]-baseline_metrics["ece"])
            trials.append({"name":name,"C":c,"features":list(cols),"score":score,**metrics})
    trials.sort(key=lambda x:(x["score"],x["brier"],x["log_loss"],len(x["features"])))
    best=trials[0]
    return best, baseline_metrics, trials[:10], cut, len(inner_train), len(inner_val)


def main():
    bat=pd.read_csv("data/mlb_batting.csv"); pit=pd.read_csv("data/mlb_pitching.csv")
    games=prepare_games(pd.read_csv("data/mlb_games.csv"))
    ph_path=Path("data/mlb_pitching_individual_history.csv"); gs_path=Path("data/mlb_game_starters_history.csv")
    if not ph_path.exists(): raise RuntimeError("Ejecuta primero build_pitcher_history.py")
    if not gs_path.exists(): raise RuntimeError("Ejecuta primero build_game_starters_history.py")
    ph=pd.read_csv(ph_path); ph["_name_key"]=ph["Name"].map(_name_key)
    starters=pd.read_csv(gs_path); starters["GameID"]=pd.to_numeric(starters["GameID"],errors="coerce")
    if "GameID" not in games.columns: raise RuntimeError("mlb_games.csv no contiene GameID para matching exacto")
    games["GameID"]=pd.to_numeric(games["GameID"],errors="coerce")
    keep=["GameID","HomeStarterID","AwayStarterID","HomeStarterName","AwayStarterName"]
    games=games.merge(starters[keep].drop_duplicates("GameID"),on="GameID",how="left")
    for old,new in (("Home_Starter","HomeStarterName"),("Away_Starter","AwayStarterName")):
        if old not in games.columns: games[old]=games[new]
        else: games[old]=games[old].where(games[old].notna() & games[old].astype(str).str.strip().ne(""),games[new])

    d1,d2=_date_boundaries(games); train=games[games["Date"].dt.normalize()<d1].copy(); later=games[games["Date"].dt.normalize()>=d1].copy()
    model=PredictorMLMLB()
    if not model.entrenar(bat,pit,train): raise RuntimeError("No se pudo entrenar baseline")
    bc=batting_metric(bat); pc=pitching_metric(pit)
    if not bc or not pc: raise RuntimeError("Métricas base no válidas")
    bd=_maps(bat,bc); pdict=_maps(pit,pc)
    bmed=float(pd.to_numeric(bat[bc],errors="coerce").median()); pmed=float(pd.to_numeric(pit[pc],errors="coerce").median())

    rows=[]
    for day,day_games in later.groupby(later["Date"].dt.normalize(),sort=True):
        pending=[]
        for _,r in day_games.iterrows():
            h=normalize_team(r["Home"]); a=normalize_team(r["Away"]); year=int(r["Season"]); sy=year-1
            pred=model.predecir_partido(h,a,float(bd.get((h,sy),bmed)),float(bd.get((a,sy),bmed)),float(pdict.get((h,sy),pmed)),float(pdict.get((a,sy),pmed)),game_date=r["Date"])
            pb=float(pred["Probabilidad_Local"])/100.0; extra,cov=_extra_features(r,bat,ph)
            hs=float(r["Home_Score"]); aw=float(r["Away_Score"])
            rows.append({"Date":r["Date"],"actual_home_win":int(hs>aw),"actual_total_runs":hs+aw,"baseline_home_prob":pb,"baseline_total_runs":float(pred["Proyeccion_Carreras"]),**cov,**extra})
            pending.append((h,a,hs,aw,r["Date"]))
        for h,a,hs,aw,gd in pending: model.actualizar_resultado(h,a,hs,aw,game_date=gd)

    frame=pd.DataFrame(rows); frame["Date"]=pd.to_datetime(frame["Date"])
    fit=frame[frame["Date"].dt.normalize()<d2].copy(); test=frame[frame["Date"].dt.normalize()>=d2].copy()
    if len(fit)<1000 or len(test)<400: raise RuntimeError(f"Muestra insuficiente fit={len(fit)} test={len(test)}")

    best,inner_base,top10,inner_cut,ntrain,nval=_select_features(fit)
    final_model=_fit_meta(fit,best["features"],best["C"])
    pcand=final_model.predict_proba(_design(test,best["features"]))[:,1]
    out=test.copy(); out["candidate_home_prob"]=pcand; out["candidate_total_runs"]=out["baseline_total_runs"]
    OUT.parent.mkdir(parents=True,exist_ok=True); out.to_csv(OUT,index=False)

    cov_cols=["home_pitcher_coverage","away_pitcher_coverage","home_platoon_coverage","away_platoon_coverage","home_exact_id","away_exact_id"]
    coverage={c:round(float(out[c].mean()),4) for c in cov_cols}
    report={
        "policy":"inner_temporal_selection_final_holdout_untouched",
        "fit_rows":len(fit),"holdout_rows":len(out),"holdout_start":str(d2.date()),
        "inner_cut":str(inner_cut.date()),"inner_train_rows":ntrain,"inner_validation_rows":nval,
        "inner_baseline":inner_base,"selected":best,"top10":top10,"coverage":coverage,
    }
    REPORT.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print("FEATURE_SEARCH",json.dumps(report,ensure_ascii=False))
    print(f"OK paired holdout: {OUT} rows={len(out)} selected={best['name']} C={best['C']} features={best['features']} coverage={coverage}")


if __name__=="__main__": main()
