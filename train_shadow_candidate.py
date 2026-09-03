"""Train the frozen shadow candidate selected by walk-forward research.

This runs only after the research Parquet has been fully enriched. It never changes
V7 prediction code or V7 picks. The resulting artifact is consumed only by the
shadow candidate path.
"""
from __future__ import annotations

from pathlib import Path
import json
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from modules.bigdata_mlb import LEGACY_ML_COLUMNS, MLBDataWarehouse
from modules.experimental_parquet import OUT

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "shadow_candidate.joblib"
META_PATH = MODEL_DIR / "shadow_candidate.json"
SHADOW_DATA_DIR = Path("data/shadow")
SHADOW_STARTER_HISTORY = SHADOW_DATA_DIR / "starter_performance_history.csv"
SOURCE_STARTER_HISTORY = Path("data/mlb_starter_performance_history.csv")

DN_METRICS = ("era", "whip", "k_pct", "bb_pct", "kbb_pct", "hr9")
DAYNIGHT_FEATURES = []
for side in ("home", "away"):
    for metric in DN_METRICS:
        DAYNIGHT_FEATURES += [f"{side}_starter_dn_{metric}", f"{side}_starter_dn_delta_{metric}"]
    DAYNIGHT_FEATURES += [f"{side}_starter_dn_ip", f"{side}_starter_dn_weight"]
TEAM_CONTEXT_FEATURES = [
    "home_ops_index", "away_ops_index", "home_wrc_plus", "away_wrc_plus",
    "home_ops_vs_l", "away_ops_vs_l", "home_ops_vs_r", "away_ops_vs_r",
    "home_team_era", "away_team_era", "home_team_xfip", "away_team_xfip",
    "home_team_whip", "away_team_whip",
]
EXTRA_FEATURES = DAYNIGHT_FEATURES + TEAM_CONTEXT_FEATURES
FEATURES = list(LEGACY_ML_COLUMNS) + EXTRA_FEATURES


def _models():
    clf = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), HistGradientBoostingClassifier(
        learning_rate=.04, max_iter=180, max_leaf_nodes=15, min_samples_leaf=30,
        l2_regularization=2.0, random_state=42))
    runs = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), HistGradientBoostingRegressor(
        learning_rate=.04, max_iter=180, max_leaf_nodes=15, min_samples_leaf=30,
        l2_regularization=2.0, random_state=42))
    diff = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), HistGradientBoostingRegressor(
        learning_rate=.04, max_iter=180, max_leaf_nodes=15, min_samples_leaf=30,
        l2_regularization=2.0, random_state=43))
    return clf, runs, diff


def _add_baseline(frame: pd.DataFrame) -> pd.DataFrame:
    bat = pd.read_csv("data/mlb_batting.csv", low_memory=False)
    pit = pd.read_csv("data/mlb_pitching.csv", low_memory=False)
    wh = MLBDataWarehouse()
    legacy = wh.legacy_ml_training_frame(bat, pit)[["game_key"] + list(LEGACY_ML_COLUMNS)].copy()
    dup = [c for c in LEGACY_ML_COLUMNS if c in frame.columns]
    return frame.drop(columns=dup, errors="ignore").merge(legacy, on="game_key", how="inner")


def main():
    if not OUT.exists():
        raise RuntimeError("Research Parquet missing")
    if not SOURCE_STARTER_HISTORY.exists():
        raise RuntimeError("Historical starter performance missing")

    frame = pd.read_parquet(OUT)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date", "target_home_win", "target_total_runs", "target_run_diff"])
    frame = frame.sort_values(["Date", "game_key"]).reset_index(drop=True)
    frame = _add_baseline(frame)
    missing = [c for c in FEATURES if c not in frame.columns]
    if missing:
        raise RuntimeError(f"Candidate features missing: {missing}")

    # Require the exact candidate groups to have enough real coverage.
    coverage = {c: float(frame[c].notna().mean()) for c in EXTRA_FEATURES}
    low = [c for c, v in coverage.items() if v < 0.50]
    if low:
        raise RuntimeError(f"Candidate feature coverage below 50%: {low}")

    X = frame[FEATURES].apply(pd.to_numeric, errors="coerce")
    yw = frame["target_home_win"].to_numpy(int)
    yr = frame["target_total_runs"].to_numpy(float)
    yd = frame["target_run_diff"].to_numpy(float)

    # Estimate live probability sigmas from a chronological final 20% validation block.
    cut = max(1000, int(len(frame) * 0.80))
    cut = min(cut, len(frame) - 250)
    vclf, vruns, vdiff = _models()
    vclf.fit(X.iloc[:cut], yw[:cut]); vruns.fit(X.iloc[:cut], yr[:cut]); vdiff.fit(X.iloc[:cut], yd[:cut])
    runs_resid = yr[cut:] - vruns.predict(X.iloc[cut:])
    diff_resid = yd[cut:] - vdiff.predict(X.iloc[cut:])
    sigma_runs = float(max(1.0, np.std(runs_resid)))
    sigma_diff = float(max(1.0, np.std(diff_resid)))

    clf, runs, diff = _models()
    clf.fit(X, yw); runs.fit(X, yr); diff.fit(X, yd)
    artifact = {
        "name": "starter_day_night+team_context",
        "version": "shadow-candidate-v1",
        "features": FEATURES,
        "extra_features": EXTRA_FEATURES,
        "classifier": clf,
        "runs_model": runs,
        "diff_model": diff,
        "sigma_runs": sigma_runs,
        "sigma_diff": sigma_diff,
        "training_rows": int(len(frame)),
        "trained_through": str(frame["Date"].max().date()),
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SHADOW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH, compress=3)
    shutil.copyfile(SOURCE_STARTER_HISTORY, SHADOW_STARTER_HISTORY)
    meta = {k: v for k, v in artifact.items() if k not in {"classifier", "runs_model", "diff_model"}}
    meta["coverage"] = {k: round(v, 4) for k, v in coverage.items()}
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
