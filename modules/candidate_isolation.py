from __future__ import annotations

from typing import Any

from .game_context import slate_date
from .pick_ledger import append_snapshot, append_shadow_snapshot
from .shadow_candidate import available as shadow_available, metadata as shadow_metadata


def _score(row: dict[str, Any]) -> float:
    try:
        return float(row.get("score", -999))
    except Exception:
        return -999.0


def _ledger_row(row: dict[str, Any], model_version: str) -> dict[str, Any]:
    away, home = str(row.get("partido") or " @ ").split(" @ ", 1)
    return {
        "game_date": slate_date().isoformat(),
        "game_pk": row.get("game_pk"),
        "away": away,
        "home": home,
        "market": row.get("mercado"),
        "selection": row.get("apuesta"),
        "line": row.get("linea"),
        "odds": row.get("cuota"),
        "prob_ml": row.get("prob_ml"),
        "prob_mc": row.get("prob_mc"),
        "prob_combined": row.get("probabilidad"),
        "market_no_vig": row.get("no_vig"),
        "edge_pp": row.get("edge_pp"),
        "ev_pct": row.get("ev_pct"),
        "disagreement_pp": row.get("desacuerdo_pp"),
        "score": row.get("score"),
        "model_version": model_version,
        "result_status": "pending",
        "kelly_pct": row.get("kelly_pct"),
    }


def scan_production(service, persist: bool = True) -> dict[str, Any]:
    """Run V7 only. This path never evaluates or writes the shadow candidate."""
    if not service.model_ready:
        raise RuntimeError("Modelo ML no disponible")

    accepted: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    games = service.slate()

    for game in games:
        try:
            result = service._evaluate_game(game)
            accepted.extend(result.get("accepted", []))
            diagnostics.extend(result.get("diagnostics", []))
        except Exception as exc:
            errors.append({
                "game_pk": game.get("game_pk"),
                "partido": f"{game.get('away')} @ {game.get('home')}",
                "error": str(exc)[:200],
            })

    accepted = sorted(accepted, key=_score, reverse=True)[:3]
    ledger_status = None
    if persist and accepted:
        rows = [_ledger_row(row, "v7-cloudrun") for row in accepted]
        try:
            append_snapshot(rows)
            ledger_status = {"ok": True, "rows": len(rows), "worksheet": "MLB_Picks"}
        except Exception as exc:
            ledger_status = {"ok": False, "rows": 0, "worksheet": "MLB_Picks", "message": str(exc)[:240]}

    return {
        "date": slate_date().isoformat(),
        "mode": "production_v7_only",
        "recommendations": accepted,
        "diagnostics": sorted(diagnostics, key=_score, reverse=True)[:12],
        "errors": errors,
        "persisted": bool(persist),
        "ledger": ledger_status,
    }


def scan_candidate(service, persist: bool = True) -> dict[str, Any]:
    """Run the shadow candidate only and write exclusively to MLB_Candidate_Picks.

    V7 is evaluated internally only to obtain the already-existing Monte Carlo market
    distribution required by the candidate scorer. No V7 recommendation is persisted.
    """
    if not service.model_ready:
        raise RuntimeError("Modelo ML no disponible")
    if not shadow_available():
        raise RuntimeError("Shadow candidate artifact unavailable")

    accepted: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    games = service.slate()

    for game in games:
        try:
            baseline_result = service._evaluate_game(game)
            shadow = service._evaluate_shadow(game, baseline_result)
            accepted.extend(shadow.get("accepted", []))
            diagnostics.extend(shadow.get("diagnostics", []))
            if shadow.get("error"):
                errors.append({
                    "game_pk": game.get("game_pk"),
                    "partido": f"{game.get('away')} @ {game.get('home')}",
                    "shadow_error": shadow.get("error"),
                })
        except Exception as exc:
            errors.append({
                "game_pk": game.get("game_pk"),
                "partido": f"{game.get('away')} @ {game.get('home')}",
                "error": str(exc)[:200],
            })

    accepted = sorted(accepted, key=_score, reverse=True)[:3]
    sheet_status = None
    if persist and accepted:
        rows = [
            _ledger_row(row, str(row.get("candidate_version") or "shadow-candidate-v1"))
            for row in accepted
        ]
        sheet_status = append_shadow_snapshot(rows, worksheet="MLB_Candidate_Picks")

    return {
        "date": slate_date().isoformat(),
        "mode": "shadow_candidate_only",
        "ready": True,
        "model": shadow_metadata(),
        "worksheet": "MLB_Candidate_Picks",
        "recommendations": accepted,
        "diagnostics": sorted(diagnostics, key=_score, reverse=True)[:12],
        "errors": errors,
        "persisted": bool(persist),
        "sheet": sheet_status,
        "production_write": False,
    }
