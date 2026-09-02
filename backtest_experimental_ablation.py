"""Exhaustive leak-safe ablation over coherent experimental MLB feature groups.

Research-only. The search uses ONLY the middle temporal selection block. The best
safe subset is frozen before it is evaluated once on the final temporal holdout.
Production code and production model files are never changed.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
import json

import numpy as np
import pandas as pd

import backtest_experimental_parquet as bt
from modules.bigdata_mlb import LEGACY_ML_COLUMNS
from modules.experimental_parquet import OUT, build_research_parquet

REPORT = Path("artifacts/experimental_ablation_backtest.json")
MIN_COVERAGE = 0.50

DAYNIGHT_METRICS = ("era", "whip", "k_pct", "bb_pct", "kbb_pct", "hr9")
DAYNIGHT_FEATURES = []
for side in ("home", "away"):
    for metric in DAYNIGHT_METRICS:
        DAYNIGHT_FEATURES.extend([
            f"{side}_starter_dn_{metric}",
            f"{side}_starter_dn_delta_{metric}",
        ])
    DAYNIGHT_FEATURES.extend([f"{side}_starter_dn_ip", f"{side}_starter_dn_weight"])

GROUPS = [
    ("starter_quality", [
        "home_starter_era", "away_starter_era",
        "home_starter_whip", "away_starter_whip",
        "home_starter_k_pct", "away_starter_k_pct",
        "home_starter_bb_pct", "away_starter_bb_pct",
        "home_starter_kbb_pct", "away_starter_kbb_pct",
        "home_starter_hr9", "away_starter_hr9",
    ]),
    ("starter_day_night", DAYNIGHT_FEATURES),
    ("bullpen_workload", [
        "home_bullpen_pitches_1d", "away_bullpen_pitches_1d",
        "home_bullpen_pitches_3d", "away_bullpen_pitches_3d",
        "home_bullpen_high_leverage_available", "away_bullpen_high_leverage_available",
    ]),
    ("team_context", [
        "home_ops_index", "away_ops_index", "home_wrc_plus", "away_wrc_plus",
        "home_ops_vs_l", "away_ops_vs_l", "home_ops_vs_r", "away_ops_vs_r",
        "home_team_era", "away_team_era", "home_team_xfip", "away_team_xfip",
        "home_team_whip", "away_team_whip",
    ]),
    ("park_rest", ["park_factor", "park_factor_hr", "home_rest_days", "away_rest_days"]),
    ("statcast_30d", [
        "home_xwoba_30d", "away_xwoba_30d", "home_xslg_30d", "away_xslg_30d",
        "home_hardhit_30d", "away_hardhit_30d", "home_barrel_30d", "away_barrel_30d",
        "home_ev_30d", "away_ev_30d",
    ]),
    ("weather", ["temperature_f", "humidity_pct", "wind_mph", "wind_out_component", "roof_closed", "day_game"]),
]


def _safe_selection(base, candidate):
    ratios = bt._ratios(base, candidate)
    composite = float(np.mean(list(ratios.values())))
    # Selection may choose any genuinely better subset, but rejects a candidate
    # that degrades any primary metric by more than 1%. The final holdout still
    # uses the original stricter +0.5% promotion gate.
    safe = composite < 1.0 and max(ratios.values()) <= 1.01
    return safe, composite, ratios


def _features_for_combo(combo, eligible_groups):
    out = []
    for name in combo:
        out.extend(eligible_groups[name])
    return list(dict.fromkeys(out))


def run_ablation():
    if not OUT.exists():
        build_research_parquet()
    frame = pd.read_parquet(OUT)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date", "target_home_win", "target_total_runs", "target_run_diff"])
    frame = frame.sort_values(["Date", "game_key"]).reset_index(drop=True)
    frame = bt._add_baseline_columns(frame)
    if len(frame) < 1500:
        raise RuntimeError(f"Histórico insuficiente para ablación robusta: {len(frame)}")

    all_group_features = list(dict.fromkeys(c for _, cols in GROUPS for c in cols))
    coverage = {c: float(frame[c].notna().mean()) for c in all_group_features if c in frame.columns}
    eligible_groups = {}
    skipped_groups = {}
    for name, cols in GROUPS:
        eligible = [c for c in cols if c in frame.columns and coverage.get(c, 0.0) >= MIN_COVERAGE]
        # Coherent groups need at least half of their intended variables present;
        # otherwise we do not pretend the concept was adequately tested.
        if len(eligible) >= max(1, int(np.ceil(len(cols) * 0.50))):
            eligible_groups[name] = eligible
        else:
            skipped_groups[name] = {
                "eligible": eligible,
                "required": max(1, int(np.ceil(len(cols) * 0.50))),
                "intended": len(cols),
            }

    names = list(eligible_groups)
    if not names:
        raise RuntimeError("No hay grupos experimentales con cobertura suficiente")

    days = pd.Series(frame["Date"].dt.normalize().unique()).sort_values().reset_index(drop=True)
    d70 = pd.Timestamp(days.iloc[int(len(days) * .70)])
    d85 = pd.Timestamp(days.iloc[int(len(days) * .85)])
    fit = frame[frame["Date"] < d70]
    select = frame[(frame["Date"] >= d70) & (frame["Date"] < d85)]
    hold = frame[frame["Date"] >= d85]
    if min(len(fit), len(select), len(hold)) < 150:
        raise RuntimeError("Split temporal demasiado pequeño")

    baseline = list(LEGACY_ML_COLUMNS)
    bsel = bt._score(fit, select, baseline)
    trials = []
    best = None

    # Exhaust every coherent group subset on SELECTION only. For seven groups
    # this is at most 127 candidate subsets, still small enough to audit.
    for size in range(1, len(names) + 1):
        for combo in combinations(names, size):
            features = _features_for_combo(combo, eligible_groups)
            score = bt._score(fit, select, baseline + features)
            safe, composite, ratios = _safe_selection(bsel, score)
            row = {
                "groups": list(combo), "feature_count": len(features), "score": score,
                "composite_ratio": composite, "improvement_pct": (1.0 - composite) * 100.0,
                "ratios": ratios, "safe": bool(safe),
            }
            trials.append(row)
            if safe and (best is None or composite < best["composite_ratio"]):
                best = row

    if best is None:
        best = {
            "groups": [], "feature_count": 0, "score": bsel,
            "composite_ratio": 1.0, "improvement_pct": 0.0,
            "ratios": {"brier": 1.0, "runs_mae": 1.0, "diff_mae": 1.0}, "safe": True,
        }

    chosen_groups = best["groups"]
    chosen_features = _features_for_combo(chosen_groups, eligible_groups)
    prehold = pd.concat([fit, select], ignore_index=True)
    base_hold = bt._score(prehold, hold, baseline)
    cand_hold = bt._score(prehold, hold, baseline + chosen_features) if chosen_features else base_hold.copy()
    promote, composite, ratios = bt._promotion(base_hold, cand_hold)

    ranked = sorted(trials, key=lambda r: r["composite_ratio"])
    payload = {
        "production_changed": False,
        "method": "exhaustive semantic-group ablation; selection-only subset search; frozen final holdout",
        "rows": int(len(frame)),
        "split": {
            "fit_end_exclusive": str(d70.date()), "holdout_start": str(d85.date()),
            "fit": len(fit), "selection": len(select), "holdout": len(hold),
        },
        "coverage_threshold": MIN_COVERAGE,
        "eligible_groups": eligible_groups,
        "skipped_groups": skipped_groups,
        "combinations_tested": len(trials),
        "selection_baseline": bsel,
        "selection_best": best,
        "selected_groups": chosen_groups,
        "selected_extra_features": chosen_features,
        "holdout_baseline": base_hold,
        "holdout_candidate": cand_hold,
        "holdout_ratios": ratios,
        "composite_ratio": composite,
        "improvement_pct": round((1.0 - composite) * 100.0, 3),
        "PROMOTE_TO_INTEGRATION_REVIEW": bool(promote and bool(chosen_features)),
        "rule": "final holdout: composite <= 0.995; no primary metric >1% worse; at least one >0.5% better",
        "top_selection_combinations": ranked[:15],
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=float))
    return payload


if __name__ == "__main__":
    run_ablation()
