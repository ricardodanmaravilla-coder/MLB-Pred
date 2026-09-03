from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from modules import multi_odds
from modules.game_context import market_from_event, match_odds_game
from modules.therundown_odds import _fetch_therundown
from modules.enriched_web_service import EnrichedMLBWebService
from modules.web_service import american_to_decimal
from modules.live_sheet_settlement import settle_pending_sheet
from modules.candidate_isolation import scan_production, scan_candidate

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="MLB Quant Analytics V7", version="7.5-hard-isolated-shadow")
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@lru_cache(maxsize=1)
def get_service() -> EnrichedMLBWebService:
    """Dedicated production V7 service instance."""
    return EnrichedMLBWebService()


@lru_cache(maxsize=1)
def get_candidate_service() -> EnrichedMLBWebService:
    """Dedicated Shadow service instance.

    Shadow intentionally does not share mutable service/cache state with V7.
    Both may read the same immutable source datasets and public odds feeds, but
    their runtime objects and ledgers are separate.
    """
    return EnrichedMLBWebService()


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    data = get_service().health()
    data["production_scan_endpoint"] = "/api/scan"
    data["candidate_scan_endpoint"] = "/api/candidate/scan"
    data["production_settle_endpoint"] = "/api/settle"
    data["candidate_settle_endpoint"] = "/api/candidate/settle"
    data["candidate_isolation"] = True
    data["candidate_service_isolated"] = get_candidate_service() is not get_service()
    return data


@app.get("/api/slate")
def slate():
    try:
        service = get_service()
        return {"date": service.health()["slate_date"], "games": service.slate(), "model": service.health()}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/odds-diagnostics")
def odds_diagnostics():
    try:
        service = get_service()
        games = service.slate()
        combined_ml = sum(1 for g in games if g.get("cuota_loc") is not None and g.get("cuota_vis") is not None)
        combined_totals = sum(1 for g in games if g.get("linea_carreras") is not None)
        combined_spreads = sum(1 for g in games if g.get("spread_loc") is not None and g.get("spread_vis") is not None)
        result = {
            "date": service.health()["slate_date"],
            "schedule_games": len(games),
            "providers": {
                "the_odds_api": bool(os.getenv("ODDS_API_KEY", "").strip() and os.getenv("ODDS_API_KEY", "").strip() != getattr(multi_odds, "_SENTINEL", "")),
                "therundown": bool(os.getenv("THERUNDOWN_KEY", "").strip()),
                "odds_api_io": bool(os.getenv("ODDS_API_IO_KEY", "").strip()),
                "sharpapi": bool(os.getenv("SHARPAPI_KEY", "").strip()),
            },
            "combined_pipeline": {
                "games_with_moneyline": combined_ml,
                "games_with_total": combined_totals,
                "games_with_spread": combined_spreads,
            },
            "therundown": {
                "normalized_events": 0,
                "matched_games": 0,
                "games_with_moneyline": 0,
                "games_with_total": 0,
                "games_with_spread": 0,
                "error": None,
            },
        }
        if result["providers"]["therundown"]:
            raw_get = getattr(multi_odds, "_ORIGINAL_GET", None)
            if raw_get is None:
                result["therundown"]["error"] = "original_http_transport_unavailable"
            else:
                events = _fetch_therundown(raw_get)
                result["therundown"]["normalized_events"] = len(events)
                for game in games:
                    event = match_odds_game(
                        events,
                        {
                            "local": game.get("home"),
                            "visita": game.get("away"),
                            "game_pk": game.get("game_pk"),
                            "start_time_utc": game.get("start_time_utc"),
                        },
                    )
                    if event is None:
                        continue
                    result["therundown"]["matched_games"] += 1
                    market = market_from_event(event, american_to_decimal)
                    if market.get("cuota_loc") is not None and market.get("cuota_vis") is not None:
                        result["therundown"]["games_with_moneyline"] += 1
                    if market.get("linea_carreras") is not None:
                        result["therundown"]["games_with_total"] += 1
                    if market.get("spread_loc") is not None and market.get("spread_vis") is not None:
                        result["therundown"]["games_with_spread"] += 1
        return result
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"odds diagnostics failed: {str(exc)[:240]}") from exc


@app.post("/api/scan")
def scan(persist: bool = True):
    """Production V7 scanner. Never evaluates or writes the shadow ledger."""
    try:
        result = scan_production(get_service(), persist=persist)
        if persist:
            result["settlement"] = settle_pending_sheet({"worksheet": "MLB_Picks"})
        return result
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/candidate/scan")
def candidate_scan(persist: bool = True):
    """Shadow scanner on its own service instance; writes only MLB_Candidate_Picks."""
    try:
        result = scan_candidate(get_candidate_service(), persist=persist)
        if persist:
            result["settlement"] = settle_pending_sheet({"worksheet": "MLB_Candidate_Picks"})
        return result
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/settle")
def settle():
    """Production settlement only."""
    result = settle_pending_sheet({"worksheet": "MLB_Picks"})
    if not result.get("ok"):
        raise HTTPException(status_code=503, detail=result)
    return result


@app.post("/api/candidate/settle")
def candidate_settle():
    """Shadow settlement only."""
    result = settle_pending_sheet({"worksheet": "MLB_Candidate_Picks"})
    if not result.get("ok"):
        raise HTTPException(status_code=503, detail=result)
    return result


@app.post("/api/reload")
def reload_model():
    try:
        production = get_service().reload()
        candidate = get_candidate_service().reload()
        return {"production": production, "candidate": candidate, "isolated": True}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web_app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")), workers=1)
