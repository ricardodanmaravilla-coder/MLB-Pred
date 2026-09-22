"""Standalone Probability Lab runner.

Runs the experiment without invoking V7 production persistence. When MLB_PROD_URL
is configured, the lab consumes V7's public /api/slate only as a read-only market
snapshot (schedule, pitchers and odds). Prediction/evaluation and persistence stay
inside the isolated Probability Lab process.
"""
import json
import os

import requests

from modules.google_sheets_ledger import sync_rows as sync_google_rows

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


def _num(value):
    try:
        return float(str(value).replace("+", "").strip())
    except (TypeError, ValueError):
        return None


def _settle_probability_lab():
    """Settle pending Probability Lab rows from official MLB game feeds only."""
    sheet_id = os.getenv("GOOGLE_SHEETS_ID", "").strip()
    if not sheet_id:
        return {"ok": False, "message": "GOOGLE_SHEETS_ID not configured", "settled": 0}

    try:
        from modules.google_sheets_ledger import _google_credentials, SHEET_HEADERS
        import gspread

        credentials, _ = _google_credentials({"sheet_id": sheet_id})
        ws = gspread.authorize(credentials).open_by_key(sheet_id).worksheet("MLB_Probability_Lab")
        values = ws.get_all_values()
        if not values:
            return {"ok": True, "settled": 0, "message": "empty sheet"}
        headers = values[0]
        pending = []
        for raw in values[1:]:
            padded = raw + [""] * max(0, len(headers) - len(raw))
            row = dict(zip(headers, padded[:len(headers)]))
            if str(row.get("result_status", "")).lower() == "pending" and row.get("game_pk"):
                pending.append(row)

        updates = []
        for row in pending:
            game_pk = str(row["game_pk"]).strip()
            feed = requests.get(
                f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live", timeout=30
            )
            feed.raise_for_status()
            payload = feed.json()
            status = payload.get("gameData", {}).get("status", {}).get("abstractGameState")
            if status != "Final":
                continue
            teams = payload.get("liveData", {}).get("linescore", {}).get("teams", {})
            away_runs = _num(teams.get("away", {}).get("runs"))
            home_runs = _num(teams.get("home", {}).get("runs"))
            if away_runs is None or home_runs is None:
                continue

            away = str(row.get("away") or "")
            home = str(row.get("home") or "")
            selection = str(row.get("selection") or "")
            market = str(row.get("market") or "").lower()
            line = _num(row.get("line")) or 0.0
            odds = _num(row.get("odds"))
            stake = _num(row.get("stake_mxn"))
            result = None

            if "hándicap" in market or "handicap" in market or "run" in market:
                selected_home = home and home.lower() in selection.lower()
                selected_away = away and away.lower() in selection.lower()
                selected_runs = home_runs if selected_home else away_runs if selected_away else None
                opponent_runs = away_runs if selected_home else home_runs if selected_away else None
                if selected_runs is not None:
                    adjusted = selected_runs + line
                    result = "win" if adjusted > opponent_runs else "loss" if adjusted < opponent_runs else "push"
            elif "moneyline" in market or market in ("ml", "ganador"):
                selected_home = home and home.lower() in selection.lower()
                selected_away = away and away.lower() in selection.lower()
                if selected_home:
                    result = "win" if home_runs > away_runs else "loss"
                elif selected_away:
                    result = "win" if away_runs > home_runs else "loss"
            elif "total" in market or "o/u" in market:
                total = away_runs + home_runs
                lower = selection.lower()
                if "over" in lower or "más" in lower:
                    result = "win" if total > line else "loss" if total < line else "push"
                elif "under" in lower or "menos" in lower:
                    result = "win" if total < line else "loss" if total > line else "push"

            if not result:
                continue

            settled = dict(row)
            settled["result_status"] = result
            settled["result_value"] = f"{int(away_runs)}-{int(home_runs)}"
            if result == "win" and odds:
                settled["profit_units"] = round(odds - 1.0, 4)
                if stake is not None:
                    settled["profit_mxn"] = round(stake * (odds - 1.0), 2)
            elif result == "loss":
                settled["profit_units"] = -1.0
                if stake is not None:
                    settled["profit_mxn"] = round(-stake, 2)
            else:
                settled["profit_units"] = 0.0
                if stake is not None:
                    settled["profit_mxn"] = 0.0
            updates.append(settled)

        status = sync_google_rows(
            updates,
            {"sheet_id": sheet_id, "worksheet": "MLB_Probability_Lab", "service_account_json": ""},
        ) if updates else {"ok": True, "updated": 0, "message": "no final pending games"}
        status["settled"] = len(updates)
        return status
    except Exception as exc:
        return {"ok": False, "settled": 0, "message": f"{type(exc).__name__}: {exc}"}


def main():
    service = EnrichedMLBWebService()
    live_games = _v7_live_slate()
    if live_games is not None:
        # Instance-only replacement: no V7 code/state is changed and the lab still
        # executes its own cloned V7 ML + Monte Carlo evaluation locally.
        service.slate = lambda: live_games

    settlement = _settle_probability_lab()
    print("Probability Lab settlement:", json.dumps(settlement, ensure_ascii=False, default=str))
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

    sheet_status = result.get("sheet_status") or {}
    if result.get("accepted") and not sheet_status.get("ok"):
        raise SystemExit("Probability Lab generated picks but failed to persist them to Google Sheets")
    if not settlement.get("ok"):
        raise SystemExit("Probability Lab settlement could not access Google Sheets")


if __name__ == "__main__":
    main()
