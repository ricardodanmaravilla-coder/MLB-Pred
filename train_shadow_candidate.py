"""Train the isolated Shadow Big Data candidate from leak-safe research data.

V7 is never imported or modified here. The model consumes a broad set of real pregame
signals, automatically excludes poorly covered experimental columns, validates on the
latest chronological block, then freezes a deployable Shadow-only artifact.
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
from sklearn.metrics import brier_score_loss, mean_absolute_error
from sklearn.pipeline import make_pipeline

from modules.bigdata_mlb import LEGACY_ML_COLUMNS, MLBDataWarehouse
from modules.experimental_parquet import OUT
from modules.shadow_bigdata_features import BIGDATA_EXTRA_FEATURES

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "shadow_candidate.joblib"
META_PATH = MODEL_DIR / "shadow_candidate.json"
SHADOW_DATA_DIR = Path("data/shadow")

SOURCE_STARTER_HISTORY = Path("data/mlb_starter_performance_history.csv")
SOURCE_BULLPEN_HISTORY = Path("data/mlb_bullpen_usage_history.csv")
SOURCE_STATCAST_DAILY = Path("data/experimental/statcast_team_daily_real.csv")
SHADOW_STARTER_HISTORY = SHADOW_DATA_DIR / "starter_performance_history.csv"
SHADOW_BULLPEN_HISTORY = SHADOW_DATA_DIR / "bullpen_usage_history.csv"
SHADOW_STATCAST_DAILY = SHADOW_DATA_DIR / "statcast_team_daily_real.csv"

MIN_EXTRA_COVERAGE = 0.35
MIN_VALIDATION_ROWS = 250


def _models():
    common = dict(
        learning_rate=.035,
        max_iter=240,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=4.0,
    )
    clf = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        HistGradientBoostingClassifier(**common, random_state=42),
    )
    runs = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        HistGradientBoostingRegressor(**common, random_state=42),
    )
    diff = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        HistGradientBoostingRegressor(**common, random_state=43),
    )
    return clf, runs, diff


def _add_baseline(frame: pd.DataFrame) -> pd.DataFrame:
    bat = pd.read_csv("data/mlb_batting.csv", low_memory=False)
    pit = pd.read_csv("data/mlb_pitching.csv", low_memory=False)
    wh = MLBDataWarehouse()
    legacy = wh.legacy_ml_training_frame(bat, pit)[["game_key"] + list(LEGACY_ML_COLUMNS)].copy()
    dup = [c for c in LEGACY_ML_COLUMNS if c in frame.columns]
    return frame.drop(columns=dup, errors="ignore").merge(legacy, on="game_key", how="inner")


def _selected_features(frame: pd.DataFrame):
    # Baseline columns are mandatory. Rich columns are admitted only when enough
    # historical pregame observations exist; missing values remain informative via
    # SimpleImputer(add_indicator=True).
    missing_base = [c for c in LEGACY_ML_COLUMNS if c not in frame.columns]
    if missing_base:
        raise RuntimeError(f"Big Data baseline features missing: {missing_base}")

    coverage = {
        c: float(frame[c].notna().mean()) if c in frame.columns else 0.0
        for c in BIGDATA_EXTRA_FEATURES
    }
    selected_extra = [c for c in BIGDATA_EXTRA_FEATURES if coverage[c] >= MIN_EXTRA_COVERAGE]
    skipped = [c for c in BIGDATA_EXTRA_FEATURES if c not in selected_extra]
    if len(selected_extra) < 20:
        raise RuntimeError(
            f"Big Data coverage insufficient: only {len(selected_extra)} rich features pass "
            f"{MIN_EXTRA_COVERAGE:.0%} coverage"
        )
    return list(LEGACY_ML_COLUMNS) + selected_extra, selected_extra, skipped, coverage


def _copy_runtime_data():
    required = {
        SOURCE_STARTER_HISTORY: SHADOW_STARTER_HISTORY,
        SOURCE_BULLPEN_HISTORY: SHADOW_BULLPEN_HISTORY,
        SOURCE_STATCAST_DAILY: SHADOW_STATCAST_DAILY,
    }
    for src in required:
        if not src.exists() or src.stat().st_size <= 0:
            raise RuntimeError(f"Required Big Data runtime source missing: {src}")
    SHADOW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for src, dst in required.items():
        shutil.copyfile(src, dst)


def main():
    if not OUT.exists():
        raise RuntimeError("Research Parquet missing")

    frame = pd.read_parquet(OUT)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date", "target_home_win", "target_total_runs", "target_run_diff"])
    frame = frame.sort_values(["Date", "game_key"]).reset_index(drop=True)
    frame = _add_baseline(frame)

    features, extra_features, skipped_features, coverage = _selected_features(frame)
    X = frame[features].apply(pd.to_numeric, errors="coerce")
    yw = frame["target_home_win"].to_numpy(int)
    yr = frame["target_total_runs"].to_numpy(float)
    yd = frame["target_run_diff"].to_numpy(float)

    cut = max(1000, int(len(frame) * 0.80))
    cut = min(cut, len(frame) - MIN_VALIDATION_ROWS)
    if cut <= 0 or len(frame) - cut < MIN_VALIDATION_ROWS:
        raise RuntimeError("Insufficient chronological validation rows")

    vclf, vruns, vdiff = _models()
    vclf.fit(X.iloc[:cut], yw[:cut])
    vruns.fit(X.iloc[:cut], yr[:cut])
    vdiff.fit(X.iloc[:cut], yd[:cut])

    vprob = vclf.predict_proba(X.iloc[cut:])[:, 1]
    vr = vruns.predict(X.iloc[cut:])
    vd = vdiff.predict(X.iloc[cut:])
    runs_resid = yr[cut:] - vr
    diff_resid = yd[cut:] - vd
    sigma_runs = float(max(1.0, np.std(runs_resid)))
    sigma_diff = float(max(1.0, np.std(diff_resid)))
    validation = {
        "rows": int(len(frame) - cut),
        "start_date": str(frame.iloc[cut]["Date"].date()),
        "end_date": str(frame.iloc[-1]["Date"].date()),
        "home_win_brier": round(float(brier_score_loss(yw[cut:], vprob)), 6),
        "total_runs_mae": round(float(mean_absolute_error(yr[cut:], vr)), 6),
        "run_diff_mae": round(float(mean_absolute_error(yd[cut:], vd)), 6),
    }

    clf, runs, diff = _models()
    clf.fit(X, yw)
    runs.fit(X, yr)
    diff.fit(X, yd)

    artifact = {
        "name": "shadow_bigdata_multisignal",
        "version": "shadow-bigdata-v3",
        "features": features,
        "extra_features": extra_features,
        "feature_count": int(len(features)),
        "candidate_feature_count": int(len(BIGDATA_EXTRA_FEATURES)),
        "classifier": clf,
        "runs_model": runs,
        "diff_model": diff,
        "sigma_runs": sigma_runs,
        "sigma_diff": sigma_diff,
        "training_rows": int(len(frame)),
        "trained_through": str(frame["Date"].max().date()),
        "validation": validation,
        "feature_policy": {
            "min_extra_coverage": MIN_EXTRA_COVERAGE,
            "missing_strategy": "median_plus_missing_indicator",
            "chronological_validation": True,
            "pregame_leak_safe": True,
        },
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH, compress=3)
    _copy_runtime_data()

    meta = {k: v for k, v in artifact.items() if k not in {"classifier", "runs_model", "diff_model"}}
    meta["coverage"] = {k: round(v, 4) for k, v in coverage.items()}
    meta["skipped_low_coverage_features"] = skipped_features
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
