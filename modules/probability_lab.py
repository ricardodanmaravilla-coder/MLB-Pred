"""Probability Lab v1 — isolated experimental selector cloned from V7 outputs.

V7 supplies the frozen evaluation structure (same inputs, ML and Monte Carlo), while
this module owns the experimental acceptance and staking rules. It never calls the V7
production writer and persists only to MLB_Probability_Lab.
"""
from copy import deepcopy

from .game_context import slate_date
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
    if out["accepted"]:
        out["probabilidad"] = round((float(out["prob_ml"]) + float(out["prob_mc"])) / 2.0, 2)
    return out


def _ledger_row(row):
    partido = str(row.get("partido") or "")
    away, home = (partido.split(" @ ", 1) + [""])[:2] if " @ " in partido else ("", "")
    return {
        "game_date": row.get("game_date") or slate_date().isoformat(),
        "game_pk": row.get("game_pk"),
        "away": away,
        "home": home,
        "market": row.get("mercado") or row.get("market"),
        "selection": row.get("apuesta") or row.get("selection"),
        "line": row.get("linea") if row.get("linea") is not None else row.get("line"),
        "odds": row.get("cuota") if row.get("cuota") is not None else row.get("odds"),
        "prob_ml": row.get("prob_ml"),
        "prob_mc": row.get("prob_mc"),
        "prob_combined": row.get("probabilidad") if row.get("probabilidad") is not None else row.get("probability"),
        "market_no_vig": row.get("no_vig") if row.get("no_vig") is not None else row.get("market_no_vig"),
        "edge_pp": row.get("edge_pp"),
        "ev_pct": row.get("ev_pct"),
        "disagreement_pp": row.get("desacuerdo_pp") if row.get("desacuerdo_pp") is not None else row.get("disagreement_pp"),
        "score": row.get("score"),
        "model_version": LAB_VERSION,
        "result_status": "pending",
        "kelly_pct": row.get("kelly_pct"),
    }


def scan_probability_lab(service, persist=True):
    """Evaluate V7 diagnostics read-only and apply only the Lab's 60/60 rule."""
    diagnostics = []
    errors = []
    games = service.slate()
    for game in games:
        try:
            result = service._evaluate_game(game)
            for row in result.get("diagnostics", []):
                lab = _lab_row(row)
                lab["game_pk"] = game.get("game_pk")
                diagnostics.append(lab)
        except Exception as exc:
            errors.append({"game_pk": game.get("game_pk"), "error": str(exc)})

    accepted = [row for row in diagnostics if row.get("accepted")]
    accepted.sort(key=lambda r: float(r.get("probabilidad") or r.get("probability") or 0.0), reverse=True)

    sheet_status = None
    if persist and accepted:
        sheet_status = append_shadow_snapshot([_ledger_row(row) for row in accepted], worksheet=LAB_WORKSHEET)

    return {
        "version": LAB_VERSION,
        "worksheet": LAB_WORKSHEET,
        "rule": "prob_ml >= 60 AND prob_mc >= 60; EV/edge informational only",
        "stake_rule": "3% at 60% joint probability, linear to 10% at 80%, capped at 10%",
        "games_seen": len(games),
        "accepted": accepted,
        "diagnostics": diagnostics,
        "errors": errors,
        "sheet_status": sheet_status,
    }
