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
from modules.web_service import MLBWebService, american_to_decimal

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="MLB Quant Analytics V7", version="7.0-cloudrun")
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@lru_cache(maxsize=1)
def get_service() -> MLBWebService:
    return MLBWebService()


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return get_service().health()


@app.get("/api/slate")
def slate():
    try:
        service = get_service()
        return {"date": service.health()["slate_date"], "games": service.slate(), "model": service.health()}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/odds-diagnostics")
def odds_diagnostics():
    """Safe production diagnostics for the MLB odds pipeline.

    Reports configuration flags and event/match counts only. It never returns
    API keys, secret values, request headers, or raw provider payloads.
    """
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
    try:
        return get_service().scan(persist=persist)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/reload")
def reload_model():
    try:
        service = get_service()
        return service.reload()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web_app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")), workers=1)
