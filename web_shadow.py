from __future__ import annotations

import os
from datetime import datetime, timezone
from functools import lru_cache

import requests
from fastapi import FastAPI, HTTPException

from modules.candidate_isolation import scan_candidate
from modules.enriched_web_service import EnrichedMLBWebService
from modules.game_context import market_from_event, match_odds_game
from modules.live_sheet_settlement import settle_pending_sheet
from modules.therundown_odds import _fetch_therundown
from modules.web_service import american_to_decimal

SHADOW_WORKSHEET = "MLB_Candidate_Picks"


class ShadowMLBWebService(EnrichedMLBWebService):
    """Shadow-only runtime with its own live odds enrichment.

    It never calls V7 endpoints and never writes the production ledger. TheRundown is
    used directly when available so the isolated service does not depend on V7's runtime
    environment. Started games are excluded to prevent late/catch-up scans from creating
    post-start recommendations.
    """

    def health(self):
        data = super().health()
        therundown = bool(os.getenv("THERUNDOWN_KEY", "").strip())
        primary = bool(os.getenv("ODDS_API_KEY", "").strip())
        data["odds_configured"] = bool(primary or therundown)
        data["shadow_odds_provider"] = "the_odds_api" if primary else "therundown" if therundown else "none"
        data["google_sheets_configured"] = bool(os.getenv("GOOGLE_SHEETS_ID", "").strip())
        return data

    @staticmethod
    def _is_future_game(game) -> bool:
        raw = game.get("start_time_utc")
        if not raw:
            return False
        try:
            start = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            return start > datetime.now(timezone.utc)
        except Exception:
            return False

    def slate(self):
        games = [g for g in super().slate() if self._is_future_game(g)]
        if not games or not os.getenv("THERUNDOWN_KEY", "").strip():
            return games

        try:
            events = _fetch_therundown(requests.get)
        except Exception:
            events = []

        if not events:
            return games

        for game in games:
            # Keep any complete primary-provider market already present. Otherwise fill
            # the missing market from Shadow's own TheRundown feed.
            if (
                game.get("cuota_loc") is not None
                and game.get("cuota_vis") is not None
                and game.get("linea_carreras") is not None
            ):
                continue
            event = match_odds_game(
                events,
                {
                    "local": game.get("home"),
                    "visita": game.get("away"),
                    "game_pk": game.get("game_pk"),
                    "start_time_utc": game.get("start_time_utc"),
                },
            )
            if event is not None:
                game.update(market_from_event(event, american_to_decimal))
        return games


app = FastAPI(title="MLB Shadow Candidate", version="1.1-shadow-only")


@lru_cache(maxsize=1)
def get_shadow_service() -> ShadowMLBWebService:
    # Dedicated process-local service instance. This app never constructs or
    # exposes the production V7 scanner/ledger endpoints.
    return ShadowMLBWebService()


@app.get("/")
def root():
    return {
        "service": "mlb-pred-shadow",
        "role": "shadow_only",
        "production_write": False,
        "worksheet": SHADOW_WORKSHEET,
    }


@app.get("/api/health")
def health():
    data = get_shadow_service().health()
    data.update({
        "service": "mlb-pred-shadow",
        "service_role": "shadow_only",
        "runtime_isolated_from_v7": True,
        "production_endpoints_exposed": False,
        "production_write": False,
        "worksheet": SHADOW_WORKSHEET,
        "scan_endpoint": "/api/candidate/scan",
        "settle_endpoint": "/api/candidate/settle",
    })
    return data


@app.post("/api/candidate/scan")
def candidate_scan(persist: bool = True):
    """Run Shadow only. The only writable ledger is MLB_Candidate_Picks."""
    try:
        result = scan_candidate(get_shadow_service(), persist=persist)
        # Defensive response invariants consumed by the scheduler.
        result["service"] = "mlb-pred-shadow"
        result["runtime_isolated_from_v7"] = True
        result["production_write"] = False
        result["worksheet"] = SHADOW_WORKSHEET
        if persist:
            result["settlement"] = settle_pending_sheet({"worksheet": SHADOW_WORKSHEET})
        return result
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/candidate/settle")
def candidate_settle():
    """Settle Shadow ledger only."""
    result = settle_pending_sheet({"worksheet": SHADOW_WORKSHEET})
    if not result.get("ok"):
        raise HTTPException(status_code=503, detail=result)
    result["service"] = "mlb-pred-shadow"
    result["production_write"] = False
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web_shadow:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")), workers=1)
