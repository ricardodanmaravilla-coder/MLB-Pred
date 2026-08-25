"""Forward audit ledger for MLB recommendations.

The local CSV remains the safe fallback. When GITHUB_TOKEN and LEDGER_GITHUB_REPO
are configured, snapshots are also persisted to the repository so Streamlit restarts
do not erase the audit trail.
"""

from pathlib import Path
from datetime import datetime, timezone
import base64
import io
import os

import pandas as pd
import requests


LEDGER_COLUMNS = [
    'snapshot_utc','game_date','game_pk','away','home','market','selection','line','odds',
    'prob_ml','prob_mc','prob_combined','market_no_vig','edge_pp','ev_pct',
    'disagreement_pp','score','starter_away','starter_home','park_factor',
    'temperature_f','wind_mph','wind_direction','model_version','result_status',
    'result_value','profit_units'
]


def _github_config():
    token = os.getenv('GITHUB_TOKEN', '').strip()
    repo = os.getenv('LEDGER_GITHUB_REPO', '').strip()
    branch = os.getenv('LEDGER_GITHUB_BRANCH', 'main').strip() or 'main'
    remote_path = os.getenv('LEDGER_GITHUB_PATH', 'data/picks_ledger.csv').strip() or 'data/picks_ledger.csv'
    return token, repo, branch, remote_path


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


def append_snapshot(rows, path='data/picks_ledger.csv'):
    if not rows:
        return 0
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    clean = []
    for row in rows:
        d = {c: row.get(c) for c in LEDGER_COLUMNS}
        d['snapshot_utc'] = d.get('snapshot_utc') or now
        d['model_version'] = d.get('model_version') or 'v5'
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
