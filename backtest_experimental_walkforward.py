"""Multi-season walk-forward robustness check for experimental MLB feature groups.

Research-only. Each target season is evaluated strictly out of sample: models train
only on games dated before that season. Candidate subsets are predefined from prior
research; this script does not search the target seasons. Production is never changed.
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

import backtest_experimental_parquet as bt
from modules.bigdata_mlb import LEGACY_ML_COLUMNS
from modules.experimental_parquet import OUT, build_research_parquet

REPORT = Path("artifacts/experimental_walkforward_backtest.json")
MIN_COVERAGE = 0.50
TARGET_SEASONS = (2024, 2025, 2026)

DAYNIGHT_METRICS = ("era", "whip", "k_pct", "bb_pct", "kbb_pct", "hr9")
DAYNIGHT_FEATURES = []
for side in ("home", "away"):
    for metric in DAYNIGHT_METRICS:
        DAYNIGHT_FEATURES += [f"{side}_starter_dn_{metric}", f"{side}_starter_dn_delta_{metric}"]
    DAYNIGHT_FEATURES += [f"{side}_starter_dn_ip", f"{side}_starter_dn_weight"]

GROUPS = {
    "starter_quality": [
        "home_starter_era", "away_starter_era", "home_starter_whip", "away_starter_whip",
        "home_starter_k_pct", "away_starter_k_pct", "home_starter_bb_pct", "away_starter_bb_pct",
        "home_starter_kbb_pct", "away_starter_kbb_pct", "home_starter_hr9", "away_starter_hr9",
    ],
    "starter_day_night": DAYNIGHT_FEATURES,
    "team_context": [
        "home_ops_index", "away_ops_index", "home_wrc_plus", "away_wrc_plus",
        "home_ops_vs_l", "away_ops_vs_l", "home_ops_vs_r", "away_ops_vs_r",
        "home_team_era", "away_team_era", "home_team_xfip", "away_team_xfip",
        "home_team_whip", "away_team_whip",
    ],
    "park_rest": ["park_factor", "park_factor_hr", "home_rest_days", "away_rest_days"],
    "statcast_30d": [
        "home_xwoba_30d", "away_xwoba_30d", "home_xslg_30d", "away_xslg_30d",
        "home_hardhit_30d", "away_hardhit_30d", "home_barrel_30d", "away_barrel_30d",
        "home_ev_30d", "away_ev_30d",
    ],
}

# Frozen candidates motivated by prior experiments; no target-season optimization.
CANDIDATES = {
    "ablation_best": ["starter_day_night", "team_context"],
    "statcast_quality_context": ["starter_quality", "team_context", "park_rest", "statcast_30d"],
    "quality_statcast": ["starter_quality", "park_rest", "statcast_30d"],
    "daynight_statcast": ["starter_quality", "starter_day_night", "park_rest", "statcast_30d"],
}


def _eligible_features(frame, groups):
    cols = []
    group_detail = {}
    for name in groups:
        intended = GROUPS[name]
        eligible = [c for c in intended if c in frame.columns and float(frame[c].notna().mean()) >= MIN_COVERAGE]
        required = max(1, int(np.ceil(len(intended) * 0.50)))
        group_detail[name] = {"eligible": eligible, "required": required, "intended": len(intended)}
        if len(eligible) < required:
            return [], group_detail
        cols.extend(eligible)
    return list(dict.fromkeys(cols)), group_detail


def _season_result(frame, season, features):
    start = pd.Timestamp(f"{season}-01-01")
    end = pd.Timestamp(f"{season + 1}-01-01")
    train = frame[frame["Date"] < start]
    test = frame[(frame["Date"] >= start) & (frame["Date"] < end)]
    if len(train) < 1000 or len(test) < 150:
        return {"available": False, "train_n": len(train), "test_n": len(test)}
    baseline = list(LEGACY_ML_COLUMNS)
    base = bt._score(train, test, baseline)
    cand = bt._score(train, test, baseline + features)
    ratios = bt._ratios(base, cand)
    composite = float(np.mean(list(ratios.values())))
    return {
        "available": True, "train_n": len(train), "test_n": len(test),
        "baseline": base, "candidate": cand, "ratios": ratios,
        "composite_ratio": composite, "improvement_pct": (1.0 - composite) * 100.0,
        "all_metrics_improved": bool(max(ratios.values()) < 1.0),
    }


def run_walkforward():
    if not OUT.exists():
        build_research_parquet()
    frame = pd.read_parquet(OUT)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date", "target_home_win", "target_total_runs", "target_run_diff"])
    frame = frame.sort_values(["Date", "game_key"]).reset_index(drop=True)
    frame = bt._add_baseline_columns(frame)

    results = {}
    for label, groups in CANDIDATES.items():
        features, detail = _eligible_features(frame, groups)
        entry = {"groups": groups, "features": features, "coverage_detail": detail, "seasons": {}}
        if not features:
            entry.update({"eligible": False, "robust": False, "reason": "insufficient_group_coverage"})
            results[label] = entry
            continue
        improvements = []
        improved_seasons = 0
        serious_degradation = False
        for season in TARGET_SEASONS:
            r = _season_result(frame, season, features)
            entry["seasons"][str(season)] = r
            if r.get("available"):
                improvements.append(r["improvement_pct"])
                if r["improvement_pct"] > 0:
                    improved_seasons += 1
                if max(r["ratios"].values()) > 1.01:
                    serious_degradation = True
        mean_imp = float(np.mean(improvements)) if improvements else 0.0
        # Deliberately stricter than the old single-holdout gate.
        robust = len(improvements) == len(TARGET_SEASONS) and mean_imp >= 0.5 and improved_seasons >= 2 and not serious_degradation
        entry.update({
            "eligible": True, "mean_improvement_pct": mean_imp,
            "improved_seasons": improved_seasons, "serious_degradation": serious_degradation,
            "robust": bool(robust),
        })
        results[label] = entry

    robust_candidates = [k for k, v in results.items() if v.get("robust")]
    payload = {
        "production_changed": False,
        "method": "season walk-forward; each target season trained only on prior dates; frozen candidate sets",
        "target_seasons": list(TARGET_SEASONS),
        "rows": len(frame),
        "criteria": "mean improvement >=0.5%; improve >=2/3 seasons; no primary metric >1% worse in any season",
        "candidates": results,
        "robust_candidates": robust_candidates,
        "PROMOTE_TO_INTEGRATION_REVIEW": bool(robust_candidates),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=float))
    return payload


if __name__ == "__main__":
    run_walkforward()
