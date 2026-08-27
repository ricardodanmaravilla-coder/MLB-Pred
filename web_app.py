from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from modules.web_service import MLBWebService

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
