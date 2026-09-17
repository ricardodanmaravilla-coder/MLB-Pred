"""Standalone Probability Lab runner.

Runs the experiment without invoking V7 production persistence. When MLB_PROD_URL
is configured, the lab consumes V7's public /api/slate only as a read-only market
snapshot (schedule, pitchers and odds). Prediction/evaluation and persistence stay
inside the isolated Probability Lab process.
"""
import json
import os

import requests

from modules.enriched_web_service import EnrichedMLBWebService
from modules.probability_lab import scan_probability_lab


def _v7_live_slate():
    base = os.getenv("MLB_PROD_URL", "").strip().rstrip("/")
    if not base:
        return None
    response = requests.get(f"{base}/api/slate", timeout=180)
    response.raise_for_status()
    payload = response.json()
    games = payload.get("games") if isinstance(payload, dict) else payload
    if not isinstance(games, list) or not games:
        raise RuntimeError("V7 public slate is empty")
    usable = [g for g in games if g.get("cuota_loc") is not None and g.get("cuota_vis") is not None and g.get("linea_carreras") is not None]
    if not usable:
        raise RuntimeError("V7 public slate has no usable live markets")
    print(f"Probability Lab read-only V7 slate: games={len(games)} usable_markets={len(usable)}")
    return games


def main():
    service = EnrichedMLBWebService()
    live_games = _v7_live_slate()
    if live_games is not None:
        # Instance-only replacement: no V7 code/state is changed and the lab still
        # executes its own cloned V7 ML + Monte Carlo evaluation locally.
        service.slate = lambda: live_games

    result = scan_probability_lab(service, persist=True)
    print(json.dumps({
        "version": result["version"],
        "worksheet": result["worksheet"],
        "rule": result["rule"],
        "stake_rule": result["stake_rule"],
        "market_source": "v7_public_slate_read_only" if live_games is not None else "local_odds_provider",
        "accepted_count": len(result["accepted"]),
        "error_count": len(result["errors"]),
        "sheet_status": result["sheet_status"],
        "accepted": result["accepted"],
        "errors": result["errors"],
    }, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
