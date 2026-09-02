"""Leak-safe backtest for the isolated experimental MLB Parquet.

This script NEVER changes production.  It compares the audited baseline feature
set against extra real pregame variables from data/experimental and writes a
promotion decision.  Only columns with sufficient historical coverage are
eligible.  Model/feature selection happens before the final temporal holdout.
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from modules.bigdata_mlb import LEGACY_ML_COLUMNS, MLBDataWarehouse
from modules.experimental_parquet import OUT, RESEARCH_FEATURES, build_research_parquet

REPORT = Path("artifacts/experimental_parquet_backtest.json")

# Never feed identity, market prices, or targets into the predictive feature
# search by accident. Market features are evaluated separately as diagnostics.
EXCLUDED_FROM_MODEL = {
    "home_starter_id","away_starter_id","home_starter_hand","away_starter_hand",
    "close_home_ml","close_away_ml","close_total","close_over_odds","close_under_odds",
    "close_home_runline","close_away_runline","market_home_no_vig","market_total_over_no_vig",
}
MIN_COVERAGE = 0.50
MAX_EXTRA_FEATURES = 40


def _models():
    clf = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        HistGradientBoostingClassifier(learning_rate=.04,max_iter=180,max_leaf_nodes=15,min_samples_leaf=30,l2_regularization=2.0,random_state=42),
    )
    reg1 = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        HistGradientBoostingRegressor(learning_rate=.04,max_iter=180,max_leaf_nodes=15,min_samples_leaf=30,l2_regularization=2.0,random_state=42),
    )
    reg2 = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        HistGradientBoostingRegressor(learning_rate=.04,max_iter=180,max_leaf_nodes=15,min_samples_leaf=30,l2_regularization=2.0,random_state=43),
    )
    return clf,reg1,reg2


def _score(train, test, cols):
    Xtr=train[cols].apply(pd.to_numeric,errors="coerce"); Xte=test[cols].apply(pd.to_numeric,errors="coerce")
    yw=train["target_home_win"].to_numpy(int); yr=train["target_total_runs"].to_numpy(float); yd=train["target_run_diff"].to_numpy(float)
    clf,runm,diffm=_models(); clf.fit(Xtr,yw); runm.fit(Xtr,yr); diffm.fit(Xtr,yd)
    p=clf.predict_proba(Xte)[:,1]; rp=runm.predict(Xte); dp=diffm.predict(Xte)
    ywt=test["target_home_win"].to_numpy(float); yrt=test["target_total_runs"].to_numpy(float); ydt=test["target_run_diff"].to_numpy(float)
    return {
        "brier":float(np.mean((p-ywt)**2)),
        "runs_mae":float(np.mean(np.abs(rp-yrt))),
        "diff_mae":float(np.mean(np.abs(dp-ydt))),
        "n":int(len(test)),
    }


def _ratios(base, candidate):
    return {
        "brier":candidate["brier"]/max(base["brier"],1e-12),
        "runs_mae":candidate["runs_mae"]/max(base["runs_mae"],1e-12),
        "diff_mae":candidate["diff_mae"]/max(base["diff_mae"],1e-12),
    }


def _promotion(base, candidate):
    r=_ratios(base,candidate); composite=float(np.mean(list(r.values())))
    # Require true aggregate improvement and forbid a material regression in any
    # primary target.  This makes promotion harder than simply winning one metric.
    promote = composite <= .995 and max(r.values()) <= 1.01 and min(r.values()) < .995
    return bool(promote), composite, r


def _add_baseline_columns(frame):
    # Reconstruct the legacy prior-season team terms already used by production.
    bat=pd.read_csv("data/mlb_batting.csv",low_memory=False); pit=pd.read_csv("data/mlb_pitching.csv",low_memory=False)
    wh=MLBDataWarehouse(); legacy=wh.legacy_ml_training_frame(bat,pit)
    legacy=legacy[["game_key"]+LEGACY_ML_COLUMNS].copy()
    dup=[c for c in LEGACY_ML_COLUMNS if c in frame.columns]
    frame=frame.drop(columns=dup,errors="ignore").merge(legacy,on="game_key",how="inner")
    return frame


def run_backtest():
    if not OUT.exists(): build_research_parquet()
    frame=pd.read_parquet(OUT)
    frame["Date"]=pd.to_datetime(frame["Date"],errors="coerce")
    frame=frame.dropna(subset=["Date","target_home_win","target_total_runs","target_run_diff"]).sort_values(["Date","game_key"]).reset_index(drop=True)
    frame=_add_baseline_columns(frame)
    if len(frame) < 1500: raise RuntimeError(f"Histórico insuficiente para gate robusto: {len(frame)}")

    coverage={c:float(frame[c].notna().mean()) for c in RESEARCH_FEATURES if c in frame.columns}
    extras=[c for c,v in coverage.items() if v>=MIN_COVERAGE and c not in EXCLUDED_FROM_MODEL and c not in LEGACY_ML_COLUMNS]
    # Deterministic coverage-first cap prevents a very wide noisy search.
    extras=sorted(extras,key=lambda c:(-coverage[c],c))[:MAX_EXTRA_FEATURES]

    days=pd.Series(frame["Date"].dt.normalize().unique()).sort_values().reset_index(drop=True)
    d70=pd.Timestamp(days.iloc[int(len(days)*.70)]); d85=pd.Timestamp(days.iloc[int(len(days)*.85)])
    fit=frame[frame["Date"]<d70]; select=frame[(frame["Date"]>=d70)&(frame["Date"]<d85)]; hold=frame[frame["Date"]>=d85]
    if min(len(fit),len(select),len(hold)) < 150: raise RuntimeError("Split temporal demasiado pequeño")

    baseline=list(LEGACY_ML_COLUMNS)
    bsel=_score(fit,select,baseline)

    # Forward coverage-ranked search in small blocks; selection sees only the
    # middle temporal block. Final holdout stays untouched until set is frozen.
    chosen=[]; best=bsel; history=[]
    for i in range(0,len(extras),5):
        trial=chosen+extras[i:i+5]
        score=_score(fit,select,baseline+trial)
        ok,comp,ratios=_promotion(bsel,score)
        history.append({"features":trial,"score":score,"composite_ratio":comp,"ratios":ratios,"accepted":ok})
        if ok:
            chosen=trial; best=score

    prehold=pd.concat([fit,select],ignore_index=True)
    base_hold=_score(prehold,hold,baseline)
    cand_hold=_score(prehold,hold,baseline+chosen) if chosen else base_hold.copy()
    promote,composite,ratios=_promotion(base_hold,cand_hold)

    payload={
        "production_changed":False,
        "rows":int(len(frame)),
        "split":{"fit_end_exclusive":str(d70.date()),"holdout_start":str(d85.date()),"fit":len(fit),"selection":len(select),"holdout":len(hold)},
        "coverage_threshold":MIN_COVERAGE,
        "eligible_extra_features":extras,
        "selected_extra_features":chosen,
        "selection_baseline":bsel,
        "selection_candidate":best,
        "holdout_baseline":base_hold,
        "holdout_candidate":cand_hold,
        "holdout_ratios":ratios,
        "composite_ratio":composite,
        "improvement_pct":round((1.0-composite)*100.0,3),
        "PROMOTE_TO_INTEGRATION_REVIEW":bool(promote and bool(chosen)),
        "rule":"composite <= 0.995; no primary metric >1% worse; at least one >0.5% better",
        "search_history":history,
    }
    REPORT.parent.mkdir(parents=True,exist_ok=True); REPORT.write_text(json.dumps(payload,indent=2,default=float),encoding="utf-8")
    print(json.dumps(payload,indent=2,default=float))
    return payload


if __name__ == "__main__":
    run_backtest()
