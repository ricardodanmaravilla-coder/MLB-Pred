"""Forward audit ledger for MLB recommendations.

The local CSV remains the safe fallback. When a GitHub token is configured, snapshots
can also be persisted to the repository so Streamlit restarts do not erase the audit trail.
Google Sheets is an optional secondary sink and can never block the primary ledger.
"""

from pathlib import Path
from datetime import datetime, timezone
import base64
import io
import os

import pandas as pd
import requests

from .google_sheets_ledger import sync_rows as sync_google_rows


DEFAULT_BANKROLL_MXN = 5000.0
LEDGER_COLUMNS = [
    'snapshot_utc','game_date','game_pk','away','home','market','selection','line','odds',
    'prob_ml','prob_mc','prob_combined','market_no_vig','edge_pp','ev_pct',
    'disagreement_pp','score','starter_away','starter_home','park_factor',
    'temperature_f','wind_mph','wind_direction','model_version','result_status',
    'result_value','profit_units','kelly_pct','bankroll_mxn','stake_mxn','profit_mxn'
]


def _secret(name, default=''):
    value = os.getenv(name, '').strip()
    if value:
        return value
    try:
        import streamlit as st
        value = st.secrets.get(name, default)
        if isinstance(value, str):
            return value.strip()
        return value
    except Exception:
        return default


def _number(value, default=None):
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        text = str(value).replace('%', '').replace('$', '').replace(',', '').strip()
        if not text:
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def bankroll_mxn():
    value = _number(_secret('BANKROLL_MXN', DEFAULT_BANKROLL_MXN), DEFAULT_BANKROLL_MXN)
    return float(value if value and value > 0 else DEFAULT_BANKROLL_MXN)


def _implied_push_probability(probability_pct, odds, ev_pct):
    """Recover push probability from push-aware EV when the scanner did not persist it.

    EV = p*(odds-1) - loss_prob = p*odds + push_prob - 1.
    This keeps Kelly identical to the scanner for integer totals/run lines while
    naturally returning zero for two-way markets and half-point lines.
    """
    p = _number(probability_pct)
    o = _number(odds)
    ev = _number(ev_pct)
    if p is None or o is None or ev is None or o <= 1:
        return 0.0
    p = max(0.0, min(1.0, p / 100.0))
    push = 1.0 + (ev / 100.0) - (p * o)
    return max(0.0, min(1.0 - p, push))


def quarter_kelly_pct(probability_pct, odds, ev_pct=None):
    p = _number(probability_pct)
    o = _number(odds)
    if p is None or o is None or o <= 1:
        return 0.0
    p = max(0.0, min(1.0, p / 100.0))
    push = _implied_push_probability(probability_pct, odds, ev_pct) if ev_pct is not None else 0.0
    q = max(0.0, 1.0 - p - push)
    decisions = p + q
    b = o - 1.0
    if decisions <= 0 or b <= 0:
        return 0.0
    full_kelly = (b * p - q) / (b * decisions)
    return round(max(0.0, full_kelly * 0.25) * 100.0, 2)


def enrich_tracking_row(row):
    """Add deterministic staking fields without changing any prediction decision."""
    d = dict(row or {})
    bank = _number(d.get('bankroll_mxn'))
    if bank is None or bank <= 0:
        bank = bankroll_mxn()
    kelly = _number(d.get('kelly_pct'))
    if kelly is None:
        kelly = quarter_kelly_pct(d.get('prob_combined'), d.get('odds'), d.get('ev_pct'))
    stake = _number(d.get('stake_mxn'))
    if stake is None:
        stake = round(bank * max(0.0, kelly) / 100.0, 2)
    d['kelly_pct'] = round(max(0.0, kelly), 2)
    d['bankroll_mxn'] = round(bank, 2)
    d['stake_mxn'] = round(max(0.0, stake), 2)
    if d.get('profit_mxn') in ('', None):
        d['profit_mxn'] = None
    return d


def _github_config():
    token = str(_secret('GITHUB_TOKEN') or '').strip()
    repo = str(_secret('LEDGER_GITHUB_REPO', 'ricardodanmaravilla-coder/MLB-Pred') or '').strip()
    branch = str(_secret('LEDGER_GITHUB_BRANCH', 'main') or 'main').strip()
    remote_path = str(_secret('LEDGER_GITHUB_PATH', 'data/picks_ledger.csv') or 'data/picks_ledger.csv').strip()
    return token, repo, branch, remote_path


def _google_config():
    return {
        'sheet_id': _secret('GOOGLE_SHEETS_ID', ''),
        'worksheet': _secret('GOOGLE_SHEETS_WORKSHEET', 'MLB_Picks'),
        'service_account_json': _secret('GOOGLE_SERVICE_ACCOUNT_JSON', ''),
    }


def persistent_backend_available():
    token, repo, _, _ = _github_config()
    return bool(token and repo and '/' in repo)


def _merge_rows(old, new):
    out = pd.concat([old, new], ignore_index=True) if not old.empty else new.copy()
    for c in LEDGER_COLUMNS:
        if c not in out.columns:
            out[c] = None
    keys = ['game_date','game_pk','away','home','market','selection','line','odds']
    out = out.drop_duplicates(subset=keys, keep='last')
    return out[LEDGER_COLUMNS]


def _remote_read():
    token, repo, branch, remote_path = _github_config()
    if not (token and repo):
        return pd.DataFrame(columns=LEDGER_COLUMNS), None
    url = f'https://api.github.com/repos/{repo}/contents/{remote_path}'
    headers = {'Authorization': f'Bearer {token}', 'Accept': 'application/vnd.github+json'}
    r = requests.get(url, headers=headers, params={'ref': branch}, timeout=15)
    if r.status_code == 404:
        return pd.DataFrame(columns=LEDGER_COLUMNS), None
    r.raise_for_status()
    payload = r.json()
    raw = base64.b64decode(payload.get('content', '')).decode('utf-8')
    df = pd.read_csv(io.StringIO(raw)) if raw.strip() else pd.DataFrame(columns=LEDGER_COLUMNS)
    return df, payload.get('sha')


def _remote_write(df, previous_sha=None):
    token, repo, branch, remote_path = _github_config()
    if not (token and repo):
        return False
    url = f'https://api.github.com/repos/{repo}/contents/{remote_path}'
    headers = {'Authorization': f'Bearer {token}', 'Accept': 'application/vnd.github+json'}
    content = base64.b64encode(df.to_csv(index=False).encode('utf-8')).decode('ascii')
    body = {'message': 'Persist MLB pick ledger snapshot', 'content': content, 'branch': branch}
    if previous_sha:
        body['sha'] = previous_sha
    r = requests.put(url, headers=headers, json=body, timeout=20)
    r.raise_for_status()
    return True


def _sync_remote(new):
    if not persistent_backend_available():
        return False
    for _ in range(2):
        old, sha = _remote_read()
        merged = _merge_rows(old, new)
        try:
            return _remote_write(merged, sha)
        except requests.HTTPError as exc:
            if exc.response is None or exc.response.status_code not in (409, 422):
                raise
    return False


def sync_google_snapshot(rows):
    """Public fail-soft helper used by scanner and settlement."""
    try:
        return sync_google_rows(rows, _google_config())
    except Exception as exc:
        return {'ok': False, 'configured': False, 'inserted': 0, 'updated': 0, 'message': str(exc)[:240]}


def _show_google_status(status):
    """Expose Sheets status in Streamlit without making the ledger depend on UI success."""
    try:
        import streamlit as st
        if not status.get('configured'):
            st.warning('Google Sheets: no configurado. Revisa GOOGLE_SHEETS_ID y GOOGLE_SERVICE_ACCOUNT_JSON en Secrets.')
        elif status.get('ok'):
            inserted = int(status.get('inserted', 0) or 0)
            updated = int(status.get('updated', 0) or 0)
            st.success(f'Google Sheets conectado: {inserted} fila(s) nueva(s), {updated} actualizada(s).')
        else:
            st.error(f"Google Sheets NO pudo guardar: {status.get('message', 'error desconocido')}")
    except Exception:
        pass


def append_snapshot(rows, path='data/picks_ledger.csv'):
    if not rows:
        return 0
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    clean = []
    for row in rows:
        source = enrich_tracking_row(row)
        d = {c: source.get(c) for c in LEDGER_COLUMNS}
        d['snapshot_utc'] = d.get('snapshot_utc') or now
        d['model_version'] = d.get('model_version') or 'v6'
        d['result_status'] = d.get('result_status') or 'pending'
        clean.append(d)
    new = pd.DataFrame(clean, columns=LEDGER_COLUMNS)
    old = load_ledger(path)
    out = _merge_rows(old, new)
    out.to_csv(p, index=False)
    try:
        _sync_remote(new)
    except Exception as exc:
        print(f'Ledger remote sync failed: {exc}')
    google_status = sync_google_snapshot(new.to_dict('records'))
    _show_google_status(google_status)
    if google_status.get('configured') and not google_status.get('ok'):
        print(f"Google Sheets sync failed: {google_status.get('message')}")
    return len(new)


def load_ledger(path='data/picks_ledger.csv'):
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    try:
        df = pd.read_csv(p)
    except Exception:
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    for c in LEDGER_COLUMNS:
        if c not in df.columns:
            df[c] = None
    return df[LEDGER_COLUMNS]
