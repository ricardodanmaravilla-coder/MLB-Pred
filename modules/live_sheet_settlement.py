from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import requests

from .google_sheets_ledger import SHEET_HEADERS, _google_credentials, _sheet_id, _worksheet_name, sync_rows


def _f(value: Any, default=None):
    try:
        if value in (None, ""): return default
        return float(str(value).replace("%", "").replace("$", "").replace(",", "").strip())
    except (TypeError, ValueError): return default


def _settle(row: dict[str, Any], home_runs: float, away_runs: float):
    market = str(row.get("market") or ""); selection = str(row.get("selection") or "")
    line = _f(row.get("line")); odds = _f(row.get("odds"))
    if odds is None or odds <= 1: return None
    result = None
    if market == "Moneyline":
        selected_home = "Local" in selection or str(row.get("home") or "") in selection
        result = "win" if ((home_runs > away_runs) if selected_home else (away_runs > home_runs)) else "loss"
    elif market == "Totales" and line is not None:
        total = home_runs + away_runs
        if math.isclose(total, line): result = "push"
        elif selection.strip().lower().startswith("over"): result = "win" if total > line else "loss"
        else: result = "win" if total < line else "loss"
    elif market == "Hándicap" and line is not None:
        selected_home = str(row.get("home") or "") in selection
        margin = (home_runs - away_runs + line) if selected_home else (away_runs - home_runs + line)
        result = "push" if math.isclose(margin, 0.0) else ("win" if margin > 0 else "loss")
    if result is None: return None
    profit_units = round(odds - 1.0, 4) if result == "win" else (-1.0 if result == "loss" else 0.0)
    stake = max(0.0, _f(row.get("stake_mxn"), 0.0) or 0.0)
    return result, profit_units, f"{int(home_runs)}-{int(away_runs)}", round(stake * profit_units, 2)


def _official_final_score(game_pk: int, timeout: int = 12):
    r = requests.get(f"https://statsapi.mlb.com/api/v1.1/game/{int(game_pk)}/feed/live", timeout=timeout)
    if r.status_code == 404: return None
    r.raise_for_status(); payload = r.json(); status = payload.get("gameData", {}).get("status", {})
    abstract = str(status.get("abstractGameState") or "").lower(); detailed = str(status.get("detailedState") or "").lower()
    if abstract != "final" and detailed not in {"final", "game over", "completed early"}: return None
    teams = payload.get("liveData", {}).get("linescore", {}).get("teams", {}); home = teams.get("home", {}).get("runs"); away = teams.get("away", {}).get("runs")
    if home is None or away is None: return None
    return float(home), float(away)


def _load_sheet_rows(config=None):
    sheet_id = _sheet_id(config)
    if not sheet_id: return [], {"ok": False, "configured": False, "message": "GOOGLE_SHEETS_ID not configured"}
    credentials, auth_source = _google_credentials(config); import gspread
    client = gspread.authorize(credentials); book = client.open_by_key(sheet_id); ws = book.worksheet(_worksheet_name(config)); values = ws.get_all_values()
    if not values: return [], {"ok": True, "configured": True, "auth_source": auth_source, "message": "empty sheet"}
    header = values[0]
    if header != SHEET_HEADERS: return [], {"ok": False, "configured": True, "auth_source": auth_source, "message": "sheet header mismatch"}
    rows = []
    for raw in values[1:]:
        padded = list(raw) + [""] * max(0, len(header) - len(raw)); rows.append(dict(zip(header, padded[:len(header)])))
    return rows, {"ok": True, "configured": True, "auth_source": auth_source, "message": "sheet loaded"}


def settle_pending_sheet(config=None, max_rows: int = 250):
    try:
        rows, status = _load_sheet_rows(config)
        if not status.get("ok"): return {**status, "checked": 0, "settled": 0}
        pending = [r for r in rows if str(r.get("result_status") or "pending").strip().lower() == "pending"][:max_rows]
        settled_rows = []; checked = 0; errors = []; score_cache = {}
        for row in pending:
            game_pk = _f(row.get("game_pk"))
            if game_pk is None: continue
            checked += 1; key = int(game_pk)
            try:
                if key not in score_cache: score_cache[key] = _official_final_score(key)
                score = score_cache[key]
            except Exception as exc:
                errors.append(f"{key}:{type(exc).__name__}"); continue
            if score is None: continue
            settled = _settle(row, score[0], score[1])
            if settled is None: continue
            result, profit_units, result_value, profit_mxn = settled; updated = dict(row)
            updated["result_status"] = result; updated["profit_units"] = profit_units; updated["result_value"] = result_value; updated["profit_mxn"] = profit_mxn; settled_rows.append(updated)
        sync_status = sync_rows(settled_rows, config) if settled_rows else {"ok": True, "configured": True, "inserted": 0, "updated": 0, "message": "nothing final yet"}
        return {"ok": bool(sync_status.get("ok")), "configured": True, "checked": checked, "settled": len(settled_rows), "updated": int(sync_status.get("updated",0) or 0),
                "pending_seen":len(pending), "errors":errors[:10], "timestamp_utc":datetime.now(timezone.utc).isoformat(), "message":sync_status.get("message","")}
    except Exception as exc:
        return {"ok":False,"configured":bool(_sheet_id(config)),"checked":0,"settled":0,"updated":0,"message":f"{type(exc).__name__}: {str(exc)}"[:500]}


def settle_all_pending_sheets(max_rows: int = 250):
    """Settle V7 and shadow candidate independently; neither can contaminate the other."""
    prod = settle_pending_sheet({"worksheet":"MLB_Picks"}, max_rows=max_rows)
    candidate = settle_pending_sheet({"worksheet":"MLB_Candidate_Picks"}, max_rows=max_rows)
    return {
        "ok": bool(prod.get("ok")) and bool(candidate.get("ok")),
        "production": prod,
        "candidate": candidate,
        "settled": int(prod.get("settled",0) or 0) + int(candidate.get("settled",0) or 0),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
