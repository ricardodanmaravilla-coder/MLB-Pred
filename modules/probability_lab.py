"""Probability Lab v1 — isolated experimental selector cloned from V7 outputs.

This module NEVER writes to the V7 production ledger. It evaluates the same V7
per-game structure, but applies an independent experimental selection rule:
both ML and Monte Carlo probabilities must be >= 60% for the same pick. EV and
edge are recorded for analysis but never used as acceptance filters.
"""
from copy import deepcopy

from .pick_ledger import append_shadow_snapshot

LAB_VERSION = "v7-probability-lab-v1"
LAB_WORKSHEET = "MLB_Probability_Lab"
MIN_MODEL_PROB = 60.0
MIN_STAKE_PCT = 3.0
MAX_STAKE_PCT = 10.0
FULL_STAKE_PROB = 80.0


def probability_stake(prob_ml, prob_mc):
    """3% at 60% joint confidence, linear to 10% at 80%, capped thereafter."""
    joint = (float(prob_ml) + float(prob_mc)) / 2.0
    if joint < MIN_MODEL_PROB:
        return 0.0
    span = FULL_STAKE_PROB - MIN_MODEL_PROB
    stake = MIN_STAKE_PCT + ((min(joint, FULL_STAKE_PROB) - MIN_MODEL_PROB) / span) * (MAX_STAKE_PCT - MIN_STAKE_PCT)
    return round(max(MIN_STAKE_PCT, min(MAX_STAKE_PCT, stake)), 2)


def _qualifies(row):
    try:
        return float(row.get("prob_ml")) >= MIN_MODEL_PROB and float(row.get("prob_mc")) >= MIN_MODEL_PROB
    except (TypeError, ValueError):
        return False


def _lab_row(row):
    out = deepcopy(row)
    out["accepted"] = _qualifies(out)
    out["reason"] = "ML y MC >= 60%" if out["accepted"] else "No cumple ML y MC >= 60%"
    out["model_version"] = LAB_VERSION
    out["filter_version"] = "probability-only-60-60-v1"
    out["kelly_pct"] = probability_stake(out.get("prob_ml"), out.get("prob_mc")) if out["accepted"] else 0.0
    out["probabilidad"] = round((float(out.get("prob_ml")) + float(out.get("prob_mc"))) / 2.0, 2) if out["accepted"] else out.get("probabilidad")
    return out


def scan_probability_lab(service, persist=True):
    """Run V7's game evaluator as a read-only clone, then apply isolated lab rules.

    The experiment does not call append_snapshot and therefore cannot write to
    MLB_Picks. Its only persistence destination is MLB_Probability_Lab.
    """
    diagnostics = []
    errors = []
    for game in service.slate():
        try:
            result = service._evaluate_game(game)
            for row in result.get("diagnostics", []):
                lab = _lab_row(row)
                lab["game_pk"] = game.get("game_pk")
                diagnostics.append(lab)
        except Exception as exc:
            errors.append({"game_pk": game.get("game_pk"), "error": str(exc)})

    accepted = [row for row in diagnostics if row.get("accepted")]
    accepted.sort(key=lambda r: float(r.get("probabilidad") or 0.0), reverse=True)

    sheet_status = None
    if persist and accepted:
        rows = []
        for row in accepted:
            partido = str(row.get("partido") or "")
            away, home = (partido.split(" @ ", 1) + [""])[:2] if " @ " in partido else ("", "")
            rows.append({
                "game_date": row.get("game_date"),
                "game_pk": row.get("game_pk"),
                "away": away,
                "home": home,
                "market": row.get("mercado"),
                "selection": row.get("apuesta"),
                "line": row.get("linea"),
                "odds_decimal": row.get("cuota"),
                "prob_ml": row.get("prob_ml"),
                "prob_mc": row.get("prob_mc"),
                "prob_final": row.get("probabilidad"),
                "no_vig": row.get("no_vig"),
                "edge_pp": row.get("edge_pp"),
                "ev_pct": row.get("ev_pct"),
                "disagreement_pp": row.get("desacuerdo_pp"),
                "score": row.get("score"),
                "model_version": LAB_VERSION,
                "filter_version": row.get("filter_version"),
                "kelly_pct": row.get("kelly_pct"),
                "status": "Pendiente",
            })
        sheet_status = append_shadow_snapshot(rows, worksheet=LAB_WORKSHEET)

    return {
        "version": LAB_VERSION,
        "worksheet": LAB_WORKSHEET,
        "rule": "prob_ml >= 60 AND prob_mc >= 60; EV/edge informational only",
        "stake_rule": "3% at 60% joint probability, linear to 10% at 80%, capped at 10%",
        "accepted": accepted,
        "diagnostics": diagnostics,
        "errors": errors,
        "sheet_status": sheet_status,
    }
