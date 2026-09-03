from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException

from modules.candidate_isolation import scan_candidate
from modules.enriched_web_service import EnrichedMLBWebService
from modules.live_sheet_settlement import settle_pending_sheet

SHADOW_WORKSHEET = "MLB_Candidate_Picks"

app = FastAPI(title="MLB Shadow Candidate", version="1.0-shadow-only")


@lru_cache(maxsize=1)
def get_shadow_service() -> EnrichedMLBWebService:
    # Dedicated process-local service instance. This app never constructs or
    # exposes the production V7 scanner/ledger endpoints.
    return EnrichedMLBWebService()


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
    import os
    import uvicorn

    uvicorn.run("web_shadow:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")), workers=1)
