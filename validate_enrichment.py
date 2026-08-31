"""Paired out-of-sample validation gate for MLB-Pred candidate enrichments.

Input CSV must contain one row per historical game with information generated strictly
before first pitch. Required columns:
  actual_home_win, baseline_home_prob, candidate_home_prob
Optional total columns:
  actual_total_runs, baseline_total_runs, candidate_total_runs

Probabilities may be expressed as 0..1 or 0..100. The script reports Brier, log loss,
ECE/calibration, paired bootstrap evidence and, when available, total-runs MAE.
It never promotes a candidate merely because accuracy or historical ROI improved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _prob(x):
    s = pd.to_numeric(x, errors="coerce").astype(float)
    if s.dropna().median() > 1.0:
        s = s / 100.0
    return s.clip(1e-6, 1 - 1e-6)


def _ece(y, p, bins=10):
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    out = 0.0
    n = len(y)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if not np.any(mask):
            continue
        out += (np.sum(mask) / n) * abs(float(np.mean(p[mask])) - float(np.mean(y[mask])))
    return float(out)


def _metrics(y, p):
    loss = (p - y) ** 2
    logloss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    pred = p >= 0.5
    return {
        "brier": float(np.mean(loss)),
        "log_loss": float(np.mean(logloss)),
        "ece": _ece(y, p),
        "accuracy": float(np.mean(pred == y)),
        "brier_losses": loss,
    }


def _bootstrap_better(base_losses, cand_losses, n_boot=4000, seed=42):
    rng = np.random.default_rng(int(seed))
    diff = np.asarray(base_losses) - np.asarray(cand_losses)
    n = len(diff)
    if n < 2:
        return {"mean_brier_gain": 0.0, "p_candidate_better": 0.0, "ci95": [0.0, 0.0]}
    draws = rng.integers(0, n, size=(int(n_boot), n))
    means = diff[draws].mean(axis=1)
    return {
        "mean_brier_gain": float(diff.mean()),
        "p_candidate_better": float(np.mean(means > 0)),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
    }


def evaluate(frame: pd.DataFrame):
    required = ["actual_home_win", "baseline_home_prob", "candidate_home_prob"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")
    x = frame.copy()
    x["y"] = pd.to_numeric(x["actual_home_win"], errors="coerce")
    x["pb"] = _prob(x["baseline_home_prob"])
    x["pc"] = _prob(x["candidate_home_prob"])
    x = x.dropna(subset=["y", "pb", "pc"])
    x = x[x["y"].isin([0, 1])]
    y = x["y"].to_numpy(float)
    pb = x["pb"].to_numpy(float)
    pc = x["pc"].to_numpy(float)
    base = _metrics(y, pb)
    cand = _metrics(y, pc)
    boot = _bootstrap_better(base.pop("brier_losses"), cand.pop("brier_losses"))

    totals = None
    total_cols = ["actual_total_runs", "baseline_total_runs", "candidate_total_runs"]
    if all(c in x.columns for c in total_cols):
        tx = x[total_cols].apply(pd.to_numeric, errors="coerce").dropna()
        if len(tx) >= 100:
            bmae = float(np.mean(np.abs(tx["baseline_total_runs"] - tx["actual_total_runs"])))
            cmae = float(np.mean(np.abs(tx["candidate_total_runs"] - tx["actual_total_runs"])))
            totals = {"rows": int(len(tx)), "baseline_mae": bmae, "candidate_mae": cmae,
                      "mae_gain": bmae - cmae}

    n = int(len(x))
    brier_gain = base["brier"] - cand["brier"]
    log_gain = base["log_loss"] - cand["log_loss"]
    calibration_ok = cand["ece"] <= base["ece"] + 0.010
    totals_ok = totals is None or totals["candidate_mae"] <= totals["baseline_mae"] * 1.005
    approved = bool(
        n >= 400
        and brier_gain >= 0.0010
        and log_gain >= 0.0005
        and calibration_ok
        and totals_ok
        and boot["p_candidate_better"] >= 0.90
    )
    reasons = []
    if n < 400: reasons.append("muestra<400")
    if brier_gain < 0.0010: reasons.append("Brier_sin_mejora_suficiente")
    if log_gain < 0.0005: reasons.append("LogLoss_sin_mejora_suficiente")
    if not calibration_ok: reasons.append("calibracion_empeora")
    if not totals_ok: reasons.append("MAE_totales_empeora")
    if boot["p_candidate_better"] < 0.90: reasons.append("evidencia_bootstrap_insuficiente")
    return {
        "rows": n,
        "baseline": base,
        "candidate": cand,
        "gains": {"brier": brier_gain, "log_loss": log_gain, "ece": base["ece"] - cand["ece"]},
        "bootstrap": boot,
        "totals": totals,
        "approved": approved,
        "decision": "PROMOTE" if approved else "KEEP_BASELINE",
        "reasons": reasons,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="CSV de predicciones históricas pareadas")
    parser.add_argument("--json-out", default="", help="Ruta opcional para guardar el reporte JSON")
    args = parser.parse_args()
    report = evaluate(pd.read_csv(args.csv))
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
