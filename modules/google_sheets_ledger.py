"""Optional Google Sheets sink for MLB scanner recommendations.

Prediction logic never depends on Google. Any auth/network/worksheet error is returned
as status and must not interrupt the scanner, ledger, settlement, Streamlit, or Cloud Run.

Authentication order:
1. GOOGLE_SERVICE_ACCOUNT_JSON / Streamlit secret when explicitly configured.
2. Google Application Default Credentials (ADC), which lets Cloud Run use its attached
   service account without storing a JSON key in environment variables.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from typing import Iterable, Mapping, Any

# Compact audit/tracking schema. Detailed pregame context (starters, park and weather)
# belongs to the prediction/research data layer, not the betting ledger. V7 no longer
# emits those six fields to Sheets, so keeping empty columns only adds noise.
LEGACY_HEADERS = [
    "record_key", "snapshot_utc", "game_date", "game_pk", "away", "home", "market",
    "selection", "line", "odds", "prob_ml", "prob_mc", "prob_combined",
    "market_no_vig", "edge_pp", "ev_pct", "disagreement_pp", "score",
    "model_version", "result_status", "result_value", "profit_units"
]
SHEET_HEADERS = LEGACY_HEADERS + ["kelly_pct", "bankroll_mxn", "stake_mxn", "profit_mxn"]
GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
]
DEFAULT_BANKROLL_MXN = 5000.0
PROTECTED_TRACKING_FIELDS = ("snapshot_utc", "bankroll_mxn", "stake_mxn")


def _clean(value: Any):
    if value is None:
        return ""
    return str(value).replace("\n", " ").strip()


def _number(value: Any, default=None):
    try:
        if value is None:
            return default
        text = str(value).replace("%", "").replace("$", "").replace(",", "").strip()
        if not text:
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def _runtime_secret(name: str, default: Any = ""):
    env_value = os.getenv(name)
    if env_value is not None and str(env_value).strip():
        return env_value
    try:
        import streamlit as st
        return st.secrets.get(name, default)
    except Exception:
        return default


def _default_bankroll() -> float:
    value = _number(_runtime_secret("BANKROLL_MXN", DEFAULT_BANKROLL_MXN), DEFAULT_BANKROLL_MXN)
    return float(value if value and value > 0 else DEFAULT_BANKROLL_MXN)


def _ensure_tracking_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    """Guarantee tracking values exist even if a caller sends a partial ledger row."""
    out = dict(row or {})
    if not _clean(out.get("snapshot_utc")):
        out["snapshot_utc"] = datetime.now(timezone.utc).isoformat()

    bankroll = _number(out.get("bankroll_mxn"))
    if bankroll is None or bankroll <= 0:
        bankroll = _default_bankroll()
        out["bankroll_mxn"] = round(bankroll, 2)

    kelly = _number(out.get("kelly_pct"), 0.0) or 0.0
    stake = _number(out.get("stake_mxn"))
    if stake is None:
        out["stake_mxn"] = round(bankroll * max(0.0, kelly) / 100.0, 2)

    return out


def record_key(row: Mapping[str, Any]) -> str:
    return "|".join([
        _clean(row.get("game_date")),
        _clean(row.get("game_pk")),
        _clean(row.get("market")),
        _clean(row.get("selection")),
        _clean(row.get("model_version")),
    ])


def market_slot_key(row: Mapping[str, Any]) -> str:
    """Stable one-pick slot per game and market.

    Selection/line/model version are intentionally excluded. Once a recommendation for
    a game+market is persisted, later scans cannot add a second wager for that same slot.
    This preserves the original recommendation instead of rewriting betting history.
    """
    return "|".join([
        _clean(row.get("game_date")),
        _clean(row.get("game_pk")),
        _clean(row.get("market")),
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


def _google_credentials(config: Mapping[str, Any] | None = None):
    """Return credentials without coupling the lab to V7 production persistence.

    GitHub Actions can provide a short-lived OAuth token produced by its existing
    Workload Identity Federation login. This avoids depending on the temporary ADC
    credential file inside the isolated runner. Cloud Run keeps using ADC normally.
    """
    payload = _credentials_payload(config)
    if payload:
        from google.oauth2.service_account import Credentials
        return Credentials.from_service_account_info(payload, scopes=GOOGLE_SCOPES), "service_account_json"

    oauth_token = os.getenv("GOOGLE_OAUTH_ACCESS_TOKEN", "").strip()
    if oauth_token:
        from google.oauth2.credentials import Credentials
        return Credentials(token=oauth_token, scopes=GOOGLE_SCOPES), "github_wif_access_token"

    import google.auth
    credentials, _ = google.auth.default(scopes=GOOGLE_SCOPES)
    return credentials, "application_default_credentials"


def configured(config: Mapping[str, Any] | None = None) -> bool:
    if not _sheet_id(config):
        return False
    try:
        _google_credentials(config)
        return True
    except Exception:
        return False


def _schema_action(values):
    """Return ok/extend/reset/fallback without mutating anything."""
    if not values:
        return "reset"
    header = values[0]
    if header == SHEET_HEADERS:
        return "ok"
    if header == LEGACY_HEADERS:
        return "extend"
    data_rows = values[1:]
    has_real_data = any(any(_clean(cell) for cell in row) for row in data_rows)
    return "fallback" if has_real_data else "reset"


def _ensure_schema(book, ws, worksheet_name, gspread):
    values = ws.get_all_values()
    action = _schema_action(values)
    if action == "ok":
        return ws, values, worksheet_name, "schema ok"
    if action == "extend":
        ws.update([SHEET_HEADERS], range_name=f"A1:{_column_letter(len(SHEET_HEADERS))}1", value_input_option="RAW")
        values[0] = SHEET_HEADERS
        return ws, values, worksheet_name, "tracking columns added"
    if action == "reset":
        ws.clear()
        ws.append_row(SHEET_HEADERS, value_input_option="RAW")
        return ws, [SHEET_HEADERS], worksheet_name, "header repaired"

    fallback_name = f"{worksheet_name}_V7"
    try:
        target = book.worksheet(fallback_name)
    except gspread.WorksheetNotFound:
        target = book.add_worksheet(title=fallback_name, rows=2000, cols=len(SHEET_HEADERS))
    fallback_values = target.get_all_values()
    fallback_action = _schema_action(fallback_values)
    if fallback_action == "fallback":
        return None, fallback_values, fallback_name, "fallback worksheet also has incompatible data"
    if fallback_action in ("reset", "extend"):
        if fallback_action == "reset":
            target.clear()
            target.append_row(SHEET_HEADERS, value_input_option="RAW")
            fallback_values = [SHEET_HEADERS]
        else:
            target.update([SHEET_HEADERS], range_name=f"A1:{_column_letter(len(SHEET_HEADERS))}1", value_input_option="RAW")
            fallback_values[0] = SHEET_HEADERS
    return target, fallback_values, fallback_name, f"using {fallback_name}; original preserved"


def _column_letter(number: int) -> str:
    letters = ""
    n = int(number)
    while n > 0:
        n, rem = divmod(n - 1, 26)
        letters = chr(65 + rem) + letters
    return letters or "A"


def sync_rows(rows: Iterable[Mapping[str, Any]], config: Mapping[str, Any] | None = None):
    """Idempotently append new rows and update existing settled rows. Never raises.

    New pending recommendations are limited to one row per game+market. Exact existing
    records can still be updated normally so settlement remains idempotent. Tracking
    fields are completed here as a final safety net so no caller can create a blank stake.
    """
    rows = [_ensure_tracking_fields(r) for r in (rows or [])]
    if not rows:
        return {"ok": True, "configured": configured(config), "inserted": 0, "updated": 0, "duplicates_skipped": 0, "message": "no rows"}

    sheet_id = _sheet_id(config)
    worksheet_name = _worksheet_name(config)
    if not sheet_id:
        return {"ok": True, "configured": False, "inserted": 0, "updated": 0, "duplicates_skipped": 0, "message": "GOOGLE_SHEETS_ID not configured"}

    try:
        credentials, auth_source = _google_credentials(config)
        import gspread

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
                "duplicates_skipped": 0, "message": schema_message, "auth_source": auth_source,
            }

        key_to_row = {}
        slot_to_row = {}
        row_cache = {}
        for idx, existing in enumerate(values[1:], start=2):
            padded = list(existing) + [""] * max(0, len(SHEET_HEADERS) - len(existing))
            existing_row = dict(zip(SHEET_HEADERS, padded[:len(SHEET_HEADERS)]))
            row_cache[idx] = existing_row
            if existing_row.get("record_key"):
                key_to_row[existing_row["record_key"]] = idx
            slot = market_slot_key(existing_row)
            if slot.strip("|"):
                slot_to_row.setdefault(slot, idx)

        append_payload = []
        update_payload = []
        duplicates_skipped = 0
        last_col = _column_letter(len(SHEET_HEADERS))
        for row in rows:
            key = record_key(row)
            enriched = dict(row)
            enriched["record_key"] = key
            existing_row_num = key_to_row.get(key)
            if existing_row_num:
                previous = row_cache.get(existing_row_num, {})
                for field in PROTECTED_TRACKING_FIELDS:
                    if not _clean(enriched.get(field)) and _clean(previous.get(field)):
                        enriched[field] = previous.get(field)
                cells = [_clean(enriched.get(h)) for h in SHEET_HEADERS]
                update_payload.append({"range": f"A{existing_row_num}:{last_col}{existing_row_num}", "values": [cells]})
                continue

            status = _clean(enriched.get("result_status") or "pending").lower()
            slot = market_slot_key(enriched)
            if status == "pending" and slot in slot_to_row:
                duplicates_skipped += 1
                continue

            cells = [_clean(enriched.get(h)) for h in SHEET_HEADERS]
            append_payload.append(cells)
            key_to_row[key] = -1
            if slot.strip("|"):
                slot_to_row.setdefault(slot, -1)

        if update_payload:
            ws.batch_update(update_payload, value_input_option="USER_ENTERED")
        if append_payload:
            ws.append_rows(append_payload, value_input_option="USER_ENTERED")

        return {
            "ok": True, "configured": True, "inserted": len(append_payload),
            "updated": len(update_payload), "duplicates_skipped": duplicates_skipped,
            "worksheet": actual_name, "message": schema_message, "auth_source": auth_source,
        }
    except Exception as exc:
        exc_type = type(exc).__name__
        exc_repr = repr(exc)
        exc_text = str(exc).strip()
        detail = exc_text or exc_repr or exc_type
        return {
            "ok": False, "configured": True, "inserted": 0, "updated": 0,
            "duplicates_skipped": 0,
            "message": f"{exc_type}: {detail}"[:500],
            "exception_type": exc_type,
            "auth_source": "unknown",
        }
