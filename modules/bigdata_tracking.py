"""Fail-soft bridge between the production pick ledger and DuckDB.

The deterministic prediction id makes scanner reruns idempotent. A settled row is
never overwritten by a later Streamlit rerun, preserving the forward audit trail.
"""

from __future__ import annotations

import hashlib
import json
from typing import Iterable

import pandas as pd

from .bigdata_mlb import FEATURE_VERSION, MLBDataWarehouse, bootstrap_from_repository
from .team_utils import normalize_team


def _num(v, default=None):
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return default
        text = str(v).replace('%', '').replace('$', '').replace(',', '').strip()
        return default if not text else float(text)
    except (TypeError, ValueError):
        return default


def _identity(row):
    parts = [
        str(row.get('game_date') or ''), str(row.get('game_pk') or ''),
        normalize_team(row.get('away')), normalize_team(row.get('home')),
        str(row.get('market') or ''), str(row.get('selection') or ''),
        str(row.get('line') or ''), str(row.get('odds') or ''),
    ]
    raw = '|'.join(parts)
    return 'ledger:' + hashlib.sha256(raw.encode('utf-8')).hexdigest()


def _game_key(row):
    pk = _num(row.get('game_pk'))
    if pk is not None:
        return f'pk:{int(pk)}'
    return f"{row.get('game_date') or ''}|{normalize_team(row.get('away'))}@{normalize_team(row.get('home'))}"


def _warehouse():
    wh = MLBDataWarehouse()
    if not wh.paths.db.exists():
        bootstrap_from_repository()
        wh = MLBDataWarehouse()
    return wh


def sync_snapshot_rows(rows: Iterable[dict]) -> dict:
    rows = [dict(r or {}) for r in (rows or [])]
    if not rows:
        return {'ok': True, 'synced': 0}
    try:
        wh = _warehouse(); con = wh.connect(); synced = 0
        try:
            wh._ensure_tracking_tables(con)
            for row in rows:
                pid = _identity(row)
                existing = con.execute('SELECT settled FROM predictions WHERE prediction_id=?', [pid]).fetchone()
                if existing and bool(existing[0]):
                    continue
                if existing:
                    con.execute('DELETE FROM predictions WHERE prediction_id=?', [pid])
                probability = _num(row.get('prob_combined'), 50.0)
                odds = _num(row.get('odds'))
                if odds is None or odds <= 1.0:
                    continue
                con.execute("""
                    INSERT INTO predictions(
                        prediction_id,created_at,game_key,game_date,home,away,market,selection,
                        prob_ml,prob_mc,probability,odds,edge_pp,ev_pct,kelly_pct,accepted,
                        model_version,feature_version,payload_json,settled
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,FALSE)
                """, [
                    pid, pd.Timestamp(row.get('snapshot_utc') or pd.Timestamp.utcnow()), _game_key(row),
                    pd.Timestamp(row.get('game_date')), normalize_team(row.get('home')), normalize_team(row.get('away')),
                    str(row.get('market') or ''), str(row.get('selection') or ''),
                    _num(row.get('prob_ml'), 50.0), _num(row.get('prob_mc'), 50.0), probability, odds,
                    _num(row.get('edge_pp')), _num(row.get('ev_pct')), _num(row.get('kelly_pct'), 0.0), True,
                    str(row.get('model_version') or 'v6'), FEATURE_VERSION, json.dumps(row, default=str),
                ])
                synced += 1
            return {'ok': True, 'synced': synced}
        finally:
            con.close()
    except Exception as exc:
        return {'ok': False, 'synced': 0, 'message': str(exc)[:240]}


def settle_snapshot_rows(rows: Iterable[dict]) -> dict:
    rows = [dict(r or {}) for r in (rows or [])]
    if not rows:
        return {'ok': True, 'settled': 0}
    sync_snapshot_rows(rows)
    try:
        wh = _warehouse(); con = wh.connect(); settled = 0
        try:
            wh._ensure_tracking_tables(con)
            for row in rows:
                status = str(row.get('result_status') or '').upper().strip()
                if status not in {'WIN', 'LOSS', 'PUSH', 'VOID'}:
                    continue
                profit = _num(row.get('profit_units'), 0.0)
                pid = _identity(row)
                con.execute("""UPDATE predictions SET settled=TRUE,result=?,profit_units=?,settled_at=CURRENT_TIMESTAMP,
                    payload_json=? WHERE prediction_id=?""",
                    [status, float(profit), json.dumps(row, default=str), pid])
                changed = con.execute('SELECT settled FROM predictions WHERE prediction_id=?', [pid]).fetchone()
                if changed and bool(changed[0]): settled += 1
            return {'ok': True, 'settled': settled}
        finally:
            con.close()
    except Exception as exc:
        return {'ok': False, 'settled': 0, 'message': str(exc)[:240]}
