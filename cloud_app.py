import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from modules.cloud_service import MLBCloudService

app = FastAPI(title="MLB Quant Analytics", version="8.0")
service = MLBCloudService()
STATIC = Path(__file__).parent / "web"
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

@app.get("/", response_class=HTMLResponse)
def home():
    return (STATIC / "index.html").read_text(encoding="utf-8")

@app.get("/api/health")
def health():
    return service.health()

@app.get("/api/games")
def games():
    return {"games": service.games()}

@app.post("/api/analyze/{game_pk}")
def analyze(game_pk: int):
    try:
        return service.analyze(game_pk)
    except KeyError:
        raise HTTPException(status_code=404, detail="Partido no encontrado")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

@app.post("/api/scan")
def scan():
    return service.scan()
