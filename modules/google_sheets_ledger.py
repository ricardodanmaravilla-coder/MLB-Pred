"""Optional Google Sheets sink for MLB scanner recommendations.

Prediction logic never depends on Google. Any auth/network/worksheet error is returned
as status and must not interrupt the scanner, ledger, settlement, or Streamlit UI.
"""
from __future__ import annotations

import json
import os
from typing import Iterable, Mapping, Any

SHEET_HEADERS = [
    "record_key", "snapshot_utc", "game_date", "game_pk", "away", "home", "market",
    "selection", "line", "odds", "prob_ml", "prob_mc", "prob_combined",
    "market_no_vig", "edge_pp", "ev_pct", "disagreement_pp", "score",
    "starter_away", "starter_home", "park_factor", "temperature_f", "wind_mph",
    "wind_direction", "model_version", "result_status", "result_value", "profit_units"
]


def _clean(value: Any):
    if value is None:
        return ""
    return str(value).replace("\n", " ").strip()


def _runtime_secret(name: str, default: Any = ""):
    env_value = os.getenv(name)
    if env_value is not None and str(env_value).strip():
        return env_value
    try:
        import streamlit as st
        return st.secrets.get(name, default)
    except Exception:
        return default


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
    raw = config.get("service_account_json")
    if raw in (None, ""):
        raw = _runtime_secret("GOOGLE_SERVICE_ACCOUNT_JSON", "")
    if isinstance(raw, Mapping):
        return dict(raw)
    raw = str(raw or "").strip()
    if not raw:
        return None
    return json.loads(raw)


def _sheet_id(config: Mapping[str, Any] | None = None) -> str:
    config = dict(config or {})
    return str(config.get("sheet_id") or _runtime_secret("GOOGLE_SHEETS_ID", "")).strip()


def _worksheet_name(config: Mapping[str, Any] | None = None) -> str:
    config = dict(config or {})
    return str(config.get("worksheet") or _runtime_secret("GOOGLE_SHEETS_WORKSHEET", "MLB_Picks")).strip() or "MLB_Picks"


def configured(config: Mapping[str, Any] | None = None) -> bool:
    if not _sheet_id(config):
        return False
    try:
        return bool(_credentials_payload(config))
    except Exception:
        return False


def _schema_action(values):
    """Return ok/reset/fallback without mutating anything.

    reset: worksheet has no real data, so replacing its header is safe.
    fallback: preserve incompatible existing data and use a dedicated V6 worksheet.
    """
    if not values:
        return "reset"
    if values[0] == SHEET_HEADERS:
        return "ok"
    data_rows = values[1:]
    has_real_data = any(any(_clean(cell) for cell in row) for row in data_rows)
    return "fallback" if has_real_data else "reset"


def _ensure_schema(book, ws, worksheet_name, gspread):
    values = ws.get_all_values()
    action = _schema_action(values)
    if action == "ok":
        return ws, values, worksheet_name, "schema ok"
    if action == "reset":
        ws.clear()
        ws.append_row(SHEET_HEADERS, value_input_option="RAW")
        return ws, [SHEET_HEADERS], worksheet_name, "header repaired"

    # Preserve any incompatible data already present. Use a stable V6 worksheet
    # so repeated scanner runs do not create a new tab every time.
    fallback_name = f"{worksheet_name}_V6"
    try:
        target = book.worksheet(fallback_name)
    except gspread.WorksheetNotFound:
        target = book.add_worksheet(title=fallback_name, rows=2000, cols=len(SHEET_HEADERS))
    fallback_values = target.get_all_values()
    fallback_action = _schema_action(fallback_values)
    if fallback_action == "fallback":
        return None, fallback_values, fallback_name, "fallback worksheet also has incompatible data"
    if fallback_action == "reset":
        target.clear()
        target.append_row(SHEET_HEADERS, value_input_option="RAW")
        fallback_values = [SHEET_HEADERS]
    return target, fallback_values, fallback_name, f"using {fallback_name}; original preserved"


def sync_rows(rows: Iterable[Mapping[str, Any]], config: Mapping[str, Any] | None = None):
    """Idempotently append new rows and update existing settled rows. Never raises."""
    rows = [dict(r) for r in rows or []]
    if not rows:
        return {"ok": True, "configured": configured(config), "inserted": 0, "updated": 0, "message": "no rows"}

    sheet_id = _sheet_id(config)
    worksheet_name = _worksheet_name(config)
    try:
        creds_payload = _credentials_payload(config)
        if not sheet_id or not creds_payload:
            return {"ok": True, "configured": False, "inserted": 0, "updated": 0, "message": "not configured"}

        import gspread
        from google.oauth2.service_account import Credentials

        credentials = Credentials.from_service_account_info(
            creds_payload,
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive.file",
            ],
        )
        client = gspread.authorize(credentials)
        book = client.open_by_key(sheet_id)
        try:
            ws = book.worksheet(worksheet_name)
        except gspread.WorksheetNotFound:
            ws = book.add_worksheet(title=worksheet_name, rows=2000, cols=len(SHEET_HEADERS))

        ws, values, actual_name, schema_message = _ensure_schema(book, ws, worksheet_name, gspread)
        if ws is None:
            return {
                "ok": False, "configured": True, "inserted": 0, "updated": 0,
                "message": schema_message
            }

        key_to_row = {}
        for idx, existing in enumerate(values[1:], start=2):
            if existing and existing[0]:
                key_to_row[existing[0]] = idx

        append_payload = []
        update_payload = []
        for row in rows:
            key = record_key(row)
            enriched = dict(row)
            enriched["record_key"] = key
            cells = [_clean(enriched.get(h)) for h in SHEET_HEADERS]
            existing_row = key_to_row.get(key)
            if existing_row:
                update_payload.append({"range": f"A{existing_row}:AB{existing_row}", "values": [cells]})
            else:
                append_payload.append(cells)
                key_to_row[key] = -1

        if update_payload:
            ws.batch_update(update_payload, value_input_option="USER_ENTERED")
        if append_payload:
            ws.append_rows(append_payload, value_input_option="USER_ENTERED")

        return {
            "ok": True, "configured": True, "inserted": len(append_payload),
            "updated": len(update_payload), "worksheet": actual_name,
            "message": schema_message
        }
    except Exception as exc:
        return {
            "ok": False, "configured": bool(sheet_id), "inserted": 0, "updated": 0,
            "message": str(exc)[:240]
        }
