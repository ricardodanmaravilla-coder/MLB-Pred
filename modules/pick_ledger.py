"""Forward audit ledger for MLB recommendations.

Historical game CSVs do not contain historical sportsbook prices. This ledger
captures the exact live recommendation state so future ROI can be measured
without reconstructing yesterday's model state from memory.
"""

from pathlib import Path
from datetime import datetime, timezone
import pandas as pd


LEDGER_COLUMNS = [
    'snapshot_utc','game_date','away','home','market','selection','line','odds',
    'prob_ml','prob_mc','prob_combined','market_no_vig','edge_pp','ev_pct',
    'disagreement_pp','score','starter_away','starter_home','park_factor',
    'temperature_f','wind_mph','wind_direction','model_version','result_status',
    'result_value','profit_units'
]


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
        d['model_version'] = d.get('model_version') or 'v3'
        d['result_status'] = d.get('result_status') or 'pending'
        clean.append(d)
    new = pd.DataFrame(clean, columns=LEDGER_COLUMNS)
    if p.exists():
        try:
            old = pd.read_csv(p)
        except Exception:
            old = pd.DataFrame(columns=LEDGER_COLUMNS)
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new

    # One exact market snapshot per game/selection/odds. This avoids Streamlit reruns
    # creating dozens of identical rows while preserving materially changed prices.
    keys = ['game_date','away','home','market','selection','line','odds']
    out = out.drop_duplicates(subset=keys, keep='last')
    out.to_csv(p, index=False)
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
