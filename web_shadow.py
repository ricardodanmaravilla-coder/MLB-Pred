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
    """Shadow-only runtime with independent compute and ledger.

    Live odds are resolved in this order: Shadow's own configured provider, direct
    TheRundown, then the production service's public /api/slate as a READ-ONLY market
    fallback. The fallback copies only market fields; it never calls V7 scan/settle and
    can never write MLB_Picks. Started games are excluded from Shadow recommendations.
    """

    MARKET_FIELDS = (
        "linea_carreras", "cuota_loc", "cuota_vis", "cuota_over", "cuota_under",
        "spread_loc", "cuota_spread_loc", "spread_vis", "cuota_spread_vis",
    )

    def health(self):
        data = super().health()
        therundown = bool(os.getenv("THERUNDOWN_KEY", "").strip())
        primary = bool(os.getenv("ODDS_API_KEY", "").strip())
        prod_fallback = bool(os.getenv("MLB_PROD_URL", "").strip())
        data["odds_configured"] = bool(primary or therundown or prod_fallback)
        data["shadow_odds_provider"] = (
            "the_odds_api" if primary else
            "therundown" if therundown else
            "production_slate_read_only" if prod_fallback else "none"
        )
        data["production_slate_read_only_fallback"] = prod_fallback
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

    @classmethod
    def _market_complete(cls, game) -> bool:
        return (
            game.get("cuota_loc") is not None
            and game.get("cuota_vis") is not None
            and game.get("linea_carreras") is not None
        )

    def _fill_from_therundown(self, games):
        if not games or not os.getenv("THERUNDOWN_KEY", "").strip():
            return
        try:
            events = _fetch_therundown(requests.get)
        except Exception:
            events = []
        if not events:
            return
        for game in games:
            if self._market_complete(game):
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

    def _fill_from_production_slate(self, games):
        base = os.getenv("MLB_PROD_URL", "").strip().rstrip("/")
        if not base or not games or all(self._market_complete(g) for g in games):
            return
        try:
            r = requests.get(f"{base}/api/slate", timeout=150)
            r.raise_for_status()
            payload = r.json()
            prod_games = payload.get("games", []) if isinstance(payload, dict) else []
        except Exception:
            return

        by_pk = {str(g.get("game_pk")): g for g in prod_games if isinstance(g, dict) and g.get("game_pk") is not None}
        for game in games:
            if self._market_complete(game):
                continue
            source = by_pk.get(str(game.get("game_pk")))
            if not source:
                continue
            for field in self.MARKET_FIELDS:
                if game.get(field) is None and source.get(field) is not None:
                    game[field] = source.get(field)

    def slate(self):
        games = [g for g in super().slate() if self._is_future_game(g)]
        self._fill_from_therundown(games)
        self._fill_from_production_slate(games)
        return games


app = FastAPI(title="MLB Shadow Candidate", version="1.2-shadow-only")


@lru_cache(maxsize=1)
def get_shadow_service() -> ShadowMLBWebService:
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
