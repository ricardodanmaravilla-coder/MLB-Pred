"""Validation-only walk-forward for the production ML + Monte Carlo blend.

Goal: test whether the current 50/50 moneyline blend is optimal without touching
production.  Every game is predicted using only information available strictly
before that game date.  Historical weather is intentionally neutral because it
is not available in the repository; the experiment is therefore about relative
ML-vs-MC weighting, not a claim that it exactly reproduces every live input.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from modules.historical_mlb import prepare_games
from modules.metric_quality import batting_metric, pitching_metric
from modules.ml_mlb import PredictorMLMLB
from modules.montecarlo_mlb import simular_partido_mlb
from modules.team_utils import normalize_team

DATA = Path("data")
ART = Path("artifacts")
ART.mkdir(exist_ok=True)

WEIGHTS = np.round(np.arange(0.0, 1.01, 0.10), 2)  # weight on ML; remainder on MC
MIN_STARTER_IP = 30.0
STARTER_LOOKBACK_DAYS = 365
MC_SIMS = 10000


def _safe_num(v, default=np.nan):
    try:
        x = float(v)
        return x if np.isfinite(x) else float(default)
    except Exception:
        return float(default)


def _ip(v):
    """Convert baseball IP notation (e.g. 5.2) to 5 + 2/3."""
    try:
        s = str(v).strip()
        if not s:
            return 0.0
        if "." not in s:
            return float(s)
        a, b = s.split(".", 1)
        outs = int(b[:1] or 0)
        if outs in (0, 1, 2):
            return float(int(a)) + outs / 3.0
        return float(s)
    except Exception:
        return 0.0


def _season_lookup(df: pd.DataFrame, metric: str):
    x = df.copy()
    x["TeamKey"] = x["Team"].map(normalize_team)
    x["Season"] = pd.to_numeric(x.get("Season"), errors="coerce")
    x[metric] = pd.to_numeric(x[metric], errors="coerce")
    x = x.dropna(subset=["TeamKey", "Season", metric])
    values = x.set_index(["TeamKey", "Season"])[metric].to_dict()
    medians = x.groupby("Season")[metric].median().to_dict()
    global_median = float(x[metric].median())
    return values, medians, global_median


def _offense_index(team, season, lookup, medians, fallback):
    year = int(season) - 1
    v = _safe_num(lookup.get((team, year)), np.nan)
    med = _safe_num(medians.get(year), fallback)
    if not np.isfinite(v) or not np.isfinite(med) or med == 0:
        return 100.0
    return float(np.clip(v / med * 100.0, 75.0, 125.0))


def _pitch_value(team, season, lookup, fallback):
    v = _safe_num(lookup.get((team, int(season) - 1)), np.nan)
    return float(v) if np.isfinite(v) and v > 0 else float(fallback)


def _park_lookup():
    p = pd.read_csv(DATA / "mlb_park_factors.csv")
    p.columns = p.columns.str.strip()
    team_col = "Team"
    pf_col = "Park_Factor" if "Park_Factor" in p.columns else "ParkFactor"
    alt_col = "Altitud" if "Altitud" in p.columns else ("AltitudeFt" if "AltitudeFt" in p.columns else None)
    out = {}
    for _, r in p.iterrows():
        key = normalize_team(r.get(team_col))
        if not key:
            continue
        pf = _safe_num(r.get(pf_col), 100.0)
        alt = _safe_num(r.get(alt_col), 0.0) if alt_col else 0.0
        out[key] = (pf, alt)
    return out


def _starter_history():
    path = DATA / "mlb_starter_quality_history.csv"
    if not path.exists():
        return pd.DataFrame()
    x = pd.read_csv(path)
    x["Date"] = pd.to_datetime(x["Date"], errors="coerce").dt.normalize()
    x["GameID"] = pd.to_numeric(x["GameID"], errors="coerce")
    x["StarterID"] = pd.to_numeric(x["StarterID"], errors="coerce")
    for c in ("H", "ER", "BB", "SO", "HR", "BF"):
        x[c] = pd.to_numeric(x.get(c), errors="coerce").fillna(0.0)
    x["IP_num"] = x["IP"].map(_ip)
    return x.dropna(subset=["Date", "GameID", "StarterID"]).sort_values("Date")


def _starter_ids_by_game(starters: pd.DataFrame):
    if starters.empty:
        return {}
    d = {}
    for _, r in starters.iterrows():
        side = str(r.get("Side", "")).strip().lower()
        if side not in ("home", "away"):
            continue
        d[(int(r.GameID), side)] = int(r.StarterID)
    return d


def _starter_prior_era(starters: pd.DataFrame, starter_id, game_date):
    if starters.empty or starter_id is None:
        return None
    start = pd.Timestamp(game_date).normalize() - pd.Timedelta(days=STARTER_LOOKBACK_DAYS)
    end = pd.Timestamp(game_date).normalize()
    q = starters[(starters["StarterID"] == int(starter_id)) & (starters["Date"] >= start) & (starters["Date"] < end)]
    if q.empty:
        return None
    ip = float(q["IP_num"].sum())
    if ip < MIN_STARTER_IP:
        return None
    er = float(q["ER"].sum())
    return float(np.clip(9.0 * er / max(ip, 1e-9), 1.5, 7.5))


def _outer_folds(games: pd.DataFrame):
    dates = np.array(sorted(games["Date"].dt.normalize().unique()))
    if len(dates) < 120:
        raise RuntimeError("Historial temporal insuficiente")
    # Expanding-window: reserve roughly the last 45% of dates for three tests.
    i0 = int(len(dates) * 0.55)
    cuts = np.linspace(i0, len(dates), 4, dtype=int)
    folds = []
    for k in range(3):
        test_dates = dates[cuts[k]:cuts[k + 1]]
        if len(test_dates) == 0:
            continue
        folds.append((k + 1, pd.Timestamp(test_dates[0]), pd.Timestamp(test_dates[-1])))
    return folds


def _metric(y, p):
    y = np.asarray(y, int); p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return float(brier_score_loss(y, p)), float(log_loss(y, p, labels=[0, 1]))


def _choose_weight(train_predictions: pd.DataFrame):
    if len(train_predictions) < 400:
        return 0.50, []
    dates = np.array(sorted(train_predictions["Date"].unique()))
    cut = pd.Timestamp(dates[max(1, int(len(dates) * 0.75)) - 1])
    val = train_predictions[train_predictions["Date"] > cut].copy()
    if len(val) < 150:
        val = train_predictions.tail(max(150, int(len(train_predictions) * 0.25))).copy()
    trials = []
    y = val["actual_home_win"].to_numpy(int)
    for w in WEIGHTS:
        p = w * val["prob_ml"].to_numpy(float) + (1.0 - w) * val["prob_mc"].to_numpy(float)
        b, ll = _metric(y, p)
        score = b + 0.15 * ll
        trials.append({"ml_weight": float(w), "brier": b, "log_loss": ll, "score": score})
    trials.sort(key=lambda z: (z["score"], abs(z["ml_weight"] - 0.50)))
    return float(trials[0]["ml_weight"]), trials


def main():
    games = prepare_games(pd.read_csv(DATA / "mlb_games.csv"))
    games = games.dropna(subset=["Date", "Home_Score", "Away_Score"]).sort_values(["Date"]).copy()
    games["Date"] = pd.to_datetime(games["Date"], errors="coerce").dt.normalize()
    games = games[games["Date"] >= pd.Timestamp("2025-01-01")].copy()
    if len(games) < 1500:
        raise RuntimeError(f"Muy pocos juegos para blend walk-forward: {len(games)}")

    bat = pd.read_csv(DATA / "mlb_batting.csv")
    pit = pd.read_csv(DATA / "mlb_pitching.csv")
    bc = batting_metric(bat); pc = pitching_metric(pit)
    if not bc or not pc:
        raise RuntimeError("No hay métricas históricas de batting/pitching válidas")
    b_lookup, b_medians, b_fallback = _season_lookup(bat, bc)
    p_lookup, _, p_fallback = _season_lookup(pit, pc)
    parks = _park_lookup()
    starters = _starter_history()
    starter_ids = _starter_ids_by_game(starters)

    folds = _outer_folds(games)
    first_test = min(x[1] for x in folds)
    initial_train = games[games["Date"] < first_test].copy()
    model = PredictorMLMLB()
    if not model.entrenar(bat, pit, initial_train):
        raise RuntimeError("No se pudo entrenar PredictorMLMLB")

    predictions = []
    # Generate all dates sequentially once; same-day results are deferred.
    for day, day_games in games.groupby("Date", sort=True):
        if day < first_test - pd.Timedelta(days=120):
            continue
        prior_games = games[games["Date"] < day]
        pending = []
        for _, r in day_games.iterrows():
            h = normalize_team(r["Home"]); a = normalize_team(r["Away"])
            if not h or not a:
                continue
            season = int(r["Season"])
            off_h = _offense_index(h, season, b_lookup, b_medians, b_fallback)
            off_a = _offense_index(a, season, b_lookup, b_medians, b_fallback)
            team_p_h = _pitch_value(h, season, p_lookup, p_fallback)
            team_p_a = _pitch_value(a, season, p_lookup, p_fallback)
            gid = int(_safe_num(r.get("GameID"), -1))
            sp_h = _starter_prior_era(starters, starter_ids.get((gid, "home")), day) or team_p_h
            sp_a = _starter_prior_era(starters, starter_ids.get((gid, "away")), day) or team_p_a
            pf, alt = parks.get(h, (100.0, 0.0))

            ml = model.predecir_partido(
                h, a,
                _safe_num(b_lookup.get((h, season - 1)), b_fallback),
                _safe_num(b_lookup.get((a, season - 1)), b_fallback),
                team_p_h, team_p_a,
                pf, game_date=day,
            )
            mc = simular_partido_mlb(
                local=h, visita=a,
                pitcher_loc_xfip=sp_h, pitcher_vis_xfip=sp_a,
                wrc_loc=off_h, wrc_vis=off_a,
                bullpen_loc_era=team_p_h, bullpen_vis_era=team_p_a,
                park_factor=pf, altitud_ft=alt,
                viento_mph=0.0, direccion_viento="None", temp_f=72.0,
                linea_carreras_casino=8.5,
                df_games=prior_games,
                num_simulaciones=MC_SIMS,
                simulation_seed=gid if gid > 0 else None,
            )
            hs = float(r["Home_Score"]); aw = float(r["Away_Score"])
            predictions.append({
                "GameID": gid, "Date": day, "fold": 0,
                "actual_home_win": int(hs > aw),
                "actual_total_runs": hs + aw,
                "prob_ml": float(ml["Probabilidad_Local"]) / 100.0,
                "prob_mc": float(mc["Moneyline"]["Gana Local"]) / 100.0,
                "ml_runs": float(ml.get("Proyeccion_Carreras", np.nan)),
                "mc_runs": float(mc.get("Carreras", {}).get("Promedio_Total", np.nan)),
                "starter_home_prior": float(sp_h), "starter_away_prior": float(sp_a),
                "starter_home_real_prior": int(starter_ids.get((gid, "home")) is not None and _starter_prior_era(starters, starter_ids.get((gid, "home")), day) is not None),
                "starter_away_real_prior": int(starter_ids.get((gid, "away")) is not None and _starter_prior_era(starters, starter_ids.get((gid, "away")), day) is not None),
            })
            pending.append((h, a, hs, aw, day))
        for h, a, hs, aw, d in pending:
            model.actualizar_resultado(h, a, hs, aw, game_date=d)

    pred = pd.DataFrame(predictions).sort_values("Date").reset_index(drop=True)
    pooled = []
    fold_reports = []
    for fold, start, end in folds:
        test = pred[(pred["Date"] >= start) & (pred["Date"] <= end)].copy()
        train = pred[pred["Date"] < start].copy()
        if len(test) < 250 or len(train) < 400:
            continue
        weight, trials = _choose_weight(train)
        test["fold"] = fold
        test["baseline_home_prob"] = 0.50 * test["prob_ml"] + 0.50 * test["prob_mc"]
        test["candidate_home_prob"] = weight * test["prob_ml"] + (1.0 - weight) * test["prob_mc"]
        test["baseline_total_runs"] = 0.50 * test["ml_runs"] + 0.50 * test["mc_runs"]
        test["candidate_total_runs"] = test["baseline_total_runs"]
        y = test["actual_home_win"].to_numpy(int)
        bb, bl = _metric(y, test["baseline_home_prob"])
        cb, cl = _metric(y, test["candidate_home_prob"])
        fold_reports.append({
            "fold": fold, "test_start": str(start.date()), "test_end": str(end.date()),
            "train_rows": int(len(train)), "test_rows": int(len(test)),
            "selected_ml_weight": weight, "selected_mc_weight": round(1.0 - weight, 2),
            "baseline_brier": bb, "candidate_brier": cb, "brier_gain": bb - cb,
            "baseline_log_loss": bl, "candidate_log_loss": cl, "logloss_gain": bl - cl,
            "inner_trials": trials,
        })
        pooled.append(test)

    if not pooled:
        raise RuntimeError("No se generaron folds válidos")
    out = pd.concat(pooled, ignore_index=True)
    coverage = float(((out["starter_home_real_prior"] == 1) & (out["starter_away_real_prior"] == 1)).mean())

    # Diagnostic fixed-weight table on pooled outer predictions (not used to select/promote).
    y = out["actual_home_win"].to_numpy(int)
    fixed = []
    for w in WEIGHTS:
        p = w * out["prob_ml"].to_numpy(float) + (1.0 - w) * out["prob_mc"].to_numpy(float)
        b, ll = _metric(y, p)
        fixed.append({"ml_weight": float(w), "mc_weight": float(1.0 - w), "brier": b, "log_loss": ll})

    report = {
        "policy": "nested_walkforward_ml_mc_blend_v1",
        "production_baseline": "50pct_ML_50pct_MC",
        "rows": int(len(out)), "dates": int(out["Date"].nunique()),
        "starter_real_prior_both_coverage": coverage,
        "historical_weather": "neutral_72F_0mph_no_direction",
        "market_total_for_moneyline_simulation": 8.5,
        "mc_simulations_per_game": MC_SIMS,
        "batting_metric": bc, "pitching_metric": pc,
        "folds": fold_reports, "fixed_weight_diagnostics_not_for_selection": fixed,
        "note": "Validation-only. Outer folds choose weight only from earlier predictions; same-day results are deferred in ML updates. No production code is changed.",
    }
    out.to_csv(ART / "ml_mc_blend_walkforward_predictions.csv", index=False)
    (ART / "ml_mc_blend_walkforward_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("ML_MC_BLEND_WALKFORWARD", json.dumps(report))
    print(f"OK artifacts/ml_mc_blend_walkforward_predictions.csv rows={len(out)} starter_prior_both={coverage:.1%}")


if __name__ == "__main__":
    main()
