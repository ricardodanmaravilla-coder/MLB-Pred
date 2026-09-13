from __future__ import annotations

from typing import Any

from .game_context import slate_date
from .pick_ledger import append_snapshot, append_shadow_snapshot, sync_google_snapshot
from .shadow_candidate import available as shadow_available, metadata as shadow_metadata


SHADOW_FILTER_VERSION = "shadow-filter-v2"
SHADOW_ALLOWED_MARKETS = {"Totales"}
SHADOW_MIN_PROB = 58.0
SHADOW_MIN_EDGE = 8.0
SHADOW_MAX_EDGE = 14.0
SHADOW_MIN_EV = 10.0
SHADOW_MAX_DISAGREEMENT = 8.0
SHADOW_KELLY_CAP = 5.0


def _score(row: dict[str, Any]) -> float:
    try:
        return float(row.get("score", -999))
    except Exception:
        return -999.0


def _diagnostic_rows(diagnostics: list[dict[str, Any]], accepted: list[dict[str, Any]], max_rejected: int = 12) -> list[dict[str, Any]]:
    accepted_keys = {
        (row.get("game_pk"), row.get("mercado"), row.get("apuesta"), row.get("linea"), row.get("cuota"))
        for row in accepted
    }
    accepted_diag, rejected_diag = [], []
    for row in diagnostics:
        key = (row.get("game_pk"), row.get("mercado"), row.get("apuesta"), row.get("linea"), row.get("cuota"))
        (accepted_diag if bool(row.get("accepted")) or key in accepted_keys else rejected_diag).append(row)
    return sorted(accepted_diag, key=_score, reverse=True) + sorted(rejected_diag, key=_score, reverse=True)[:max_rejected]


def _shadow_filter(row: dict[str, Any]) -> tuple[bool, str]:
    """Precision gate kept constant while the underlying Shadow model evolves."""
    failures: list[str] = []
    if not bool(row.get("accepted")):
        failures.append(str(row.get("reason") or "base shadow filter"))
    market = str(row.get("mercado") or "")
    if market not in SHADOW_ALLOWED_MARKETS:
        failures.append("shadow: mercado pausado")

    def f(name, default):
        try: return float(row.get(name))
        except Exception: return default

    prob = f("probabilidad", -999.0)
    edge = f("edge_pp", -999.0)
    ev = f("ev_pct", -999.0)
    disagreement = f("desacuerdo_pp", 999.0)
    if prob < SHADOW_MIN_PROB: failures.append(f"shadow: prob<{SHADOW_MIN_PROB:.0f}%")
    if edge < SHADOW_MIN_EDGE: failures.append(f"shadow: edge<{SHADOW_MIN_EDGE:.0f}pp")
    if edge > SHADOW_MAX_EDGE: failures.append(f"shadow: edge>{SHADOW_MAX_EDGE:.0f}pp")
    if ev < SHADOW_MIN_EV: failures.append(f"shadow: EV<{SHADOW_MIN_EV:.0f}%")
    if disagreement > SHADOW_MAX_DISAGREEMENT: failures.append(f"shadow: desacuerdo>{SHADOW_MAX_DISAGREEMENT:.0f}pp")
    return not failures, "Cumple filtros Shadow" if not failures else "; ".join(failures)


def _apply_shadow_filter(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    accepted, reason = _shadow_filter(out)
    out["accepted"] = accepted
    out["reason"] = reason
    out["filter_version"] = SHADOW_FILTER_VERSION
    # Preserve candidate_version emitted by the actual model (e.g. shadow-bigdata-v3).
    try:
        raw_kelly = max(0.0, float(out.get("kelly_pct") or 0.0))
        out["kelly_raw_pct"] = raw_kelly
        out["kelly_pct"] = min(raw_kelly, SHADOW_KELLY_CAP)
    except Exception:
        pass
    return out


def _ledger_row(row: dict[str, Any], model_version: str) -> dict[str, Any]:
    away, home = str(row.get("partido") or " @ ").split(" @ ", 1)
    return {
        "game_date": slate_date().isoformat(), "game_pk": row.get("game_pk"),
        "away": away, "home": home, "market": row.get("mercado"), "selection": row.get("apuesta"),
        "line": row.get("linea"), "odds": row.get("cuota"), "prob_ml": row.get("prob_ml"),
        "prob_mc": row.get("prob_mc"), "prob_combined": row.get("probabilidad"),
        "market_no_vig": row.get("no_vig"), "edge_pp": row.get("edge_pp"), "ev_pct": row.get("ev_pct"),
        "disagreement_pp": row.get("desacuerdo_pp"), "score": row.get("score"),
        "model_version": model_version, "result_status": "pending", "kelly_pct": row.get("kelly_pct"),
    }


def scan_production(service, persist: bool = True) -> dict[str, Any]:
    """Run V7 only. This path never evaluates or writes the shadow candidate."""
    if not service.model_ready: raise RuntimeError("Modelo ML no disponible")
    accepted, diagnostics, errors = [], [], []
    games = service.slate()
    for game in games:
        try:
            result = service._evaluate_game(game)
            accepted.extend(result.get("accepted", [])); diagnostics.extend(result.get("diagnostics", []))
        except Exception as exc:
            errors.append({"game_pk": game.get("game_pk"), "partido": f"{game.get('away')} @ {game.get('home')}", "error": str(exc)[:200]})
    accepted = sorted(accepted, key=_score, reverse=True)
    ledger_status = None
    if persist and accepted:
        rows = [_ledger_row(row, "v7-cloudrun") for row in accepted]
        try:
            append_snapshot(rows)
            google_status = sync_google_snapshot(rows, worksheet="MLB_Picks")
            ledger_status = {"ok": bool(google_status.get("ok")), "rows": len(rows), "worksheet": google_status.get("worksheet") or "MLB_Picks", "google_sheets": google_status}
        except Exception as exc:
            ledger_status = {"ok": False, "rows": 0, "worksheet": "MLB_Picks", "message": str(exc)[:240]}
    return {"date": slate_date().isoformat(), "mode": "production_v7_only", "games_seen": len(games), "recommendations": accepted,
            "diagnostics": _diagnostic_rows(diagnostics, accepted), "errors": errors, "persisted": bool(persist), "ledger": ledger_status}


def scan_candidate(service, persist: bool = True) -> dict[str, Any]:
    """Run isolated Shadow and write exclusively to MLB_Candidate_Picks."""
    if not service.model_ready: raise RuntimeError("Modelo ML no disponible")
    if not shadow_available(): raise RuntimeError("Shadow candidate artifact unavailable")

    accepted, diagnostics, errors = [], [], []
    games = service.slate()
    for game in games:
        try:
            baseline_result = service._evaluate_game(game)
            shadow = service._evaluate_shadow(game, baseline_result)
            filtered = [_apply_shadow_filter(row) for row in shadow.get("diagnostics", [])]
            diagnostics.extend(filtered); accepted.extend(row for row in filtered if row.get("accepted"))
            if shadow.get("error"):
                errors.append({"game_pk": game.get("game_pk"), "partido": f"{game.get('away')} @ {game.get('home')}", "shadow_error": shadow.get("error")})
        except Exception as exc:
            errors.append({"game_pk": game.get("game_pk"), "partido": f"{game.get('away')} @ {game.get('home')}", "error": str(exc)[:200]})

    accepted = sorted(accepted, key=_score, reverse=True)
    meta = dict(shadow_metadata() or {})
    active_model_version = str(meta.get("version") or "shadow-candidate")
    sheet_status = None
    if persist and accepted:
        rows = [_ledger_row(row, str(row.get("candidate_version") or active_model_version)) for row in accepted]
        sheet_status = append_shadow_snapshot(rows, worksheet="MLB_Candidate_Picks")

    meta.update({
        "filter_version": SHADOW_FILTER_VERSION,
        "markets": sorted(SHADOW_ALLOWED_MARKETS),
        "min_probability": SHADOW_MIN_PROB,
        "edge_range_pp": [SHADOW_MIN_EDGE, SHADOW_MAX_EDGE],
        "min_ev_pct": SHADOW_MIN_EV,
        "max_disagreement_pp": SHADOW_MAX_DISAGREEMENT,
        "kelly_cap_pct": SHADOW_KELLY_CAP,
    })
    return {
        "date": slate_date().isoformat(), "mode": "shadow_candidate_only", "games_seen": len(games), "ready": True,
        "model": meta, "worksheet": "MLB_Candidate_Picks", "recommendations": accepted,
        "diagnostics": _diagnostic_rows(diagnostics, accepted), "errors": errors, "persisted": bool(persist),
        "sheet": sheet_status, "production_write": False,
    }
