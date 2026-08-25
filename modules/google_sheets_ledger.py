"""Optional Google Sheets sink for scanner recommendations.

This module is deliberately fail-soft and side-effect isolated: prediction logic never
imports Google libraries directly, and a Sheets outage/configuration problem must never
prevent the scanner or primary ledger from working.
"""
from __future__ import annotations

import json
import os
from typing import Iterable, Mapping, Any

SHEET_HEADERS = [
    "record_key", "game_date", "game_pk", "away", "home", "market", "selection",
    "line", "odds", "prob_ml", "prob_mc", "prob_combined", "market_no_vig",
    "edge_pp", "ev_pct", "disagreement_pp", "score", "starter_away",
    "starter_home", "park_factor", "temperature_f", "wind_mph", "wind_direction",
    "model_version", "result_status", "result_home", "result_away",
    "profit_units", "roi_pct"
]


def _clean(value: Any):
    if value is None:
        return ""
    return str(value).replace("\n", " ").strip()


def record_key(row: Mapping[str, Any]) -> str:
    return "|".join([
        _clean(row.get("game_date")),
        _clean(row.get("game_pk")),
        _clean(row.get("market")),
        _clean(row.get("selection")),
        _clean(row.get("model_version")),
    ])


def _credentials_payload(config: Mapping[str, Any] | None = None):
    config = dict(config or {})
    raw = config.get("service_account_json") or os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "")
    if isinstance(raw, Mapping):
        return dict(raw)
    raw = str(raw or "").strip()
    if not raw:
        return None
    return json.loads(raw)


def configured(config: Mapping[str, Any] | None = None) -> bool:
    config = dict(config or {})
    sheet_id = str(config.get("sheet_id") or os.getenv("GOOGLE_SHEETS_ID", "")).strip()
    if not sheet_id:
        return False
    try:
        return bool(_credentials_payload(config))
    except Exception:
        return False


def append_recommendations(rows: Iterable[Mapping[str, Any]], config: Mapping[str, Any] | None = None):
    """Append only unseen recommendation rows. Returns a status dict; never raises."""
    rows = [dict(r) for r in rows or []]
    if not rows:
        return {"ok": True, "configured": configured(config), "appended": 0, "skipped": 0, "message": "no rows"}

    config = dict(config or {})
    sheet_id = str(config.get("sheet_id") or os.getenv("GOOGLE_SHEETS_ID", "")).strip()
    worksheet_name = str(config.get("worksheet") or os.getenv("GOOGLE_SHEETS_WORKSHEET", "MLB_Picks")).strip() or "MLB_Picks"
    try:
        creds_payload = _credentials_payload(config)
        if not sheet_id or not creds_payload:
            return {"ok": True, "configured": False, "appended": 0, "skipped": len(rows), "message": "not configured"}

        import gspread
        from google.oauth2.service_account import Credentials

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive.file",
        ]
        credentials = Credentials.from_service_account_info(creds_payload, scopes=scopes)
        client = gspread.authorize(credentials)
        book = client.open_by_key(sheet_id)
        try:
            ws = book.worksheet(worksheet_name)
        except gspread.WorksheetNotFound:
            ws = book.add_worksheet(title=worksheet_name, rows=2000, cols=len(SHEET_HEADERS))

        existing = ws.row_values(1)
        if not existing:
            ws.append_row(SHEET_HEADERS, value_input_option="RAW")
        elif existing != SHEET_HEADERS:
            return {"ok": False, "configured": True, "appended": 0, "skipped": len(rows), "message": "worksheet header mismatch"}

        existing_keys = set(filter(None, ws.col_values(1)[1:]))
        payload = []
        skipped = 0
        for row in rows:
            key = record_key(row)
            if key in existing_keys:
                skipped += 1
                continue
            enriched = dict(row)
            enriched["record_key"] = key
            payload.append([_clean(enriched.get(h)) for h in SHEET_HEADERS])
            existing_keys.add(key)

        if payload:
            ws.append_rows(payload, value_input_option="USER_ENTERED")
        return {"ok": True, "configured": True, "appended": len(payload), "skipped": skipped, "message": "ok"}
    except Exception as exc:
        return {"ok": False, "configured": bool(sheet_id), "appended": 0, "skipped": len(rows), "message": str(exc)[:240]}
