import json

import pandas as pd

from modules.pick_ledger import load_ledger


def _summary(df):
    if df.empty:
        return {'n': 0, 'decisions': 0, 'wins': 0, 'losses': 0, 'pushes': 0, 'hit_rate': None, 'profit_units': 0.0, 'roi_per_bet': None}
    status = df['result_status'].astype(str)
    wins = int((status == 'win').sum())
    losses = int((status == 'loss').sum())
    pushes = int((status == 'push').sum())
    decisions = wins + losses
    profit = float(pd.to_numeric(df['profit_units'], errors='coerce').fillna(0).sum())
    n = int(len(df))
    return {
        'n': n,
        'decisions': decisions,
        'wins': wins,
        'losses': losses,
        'pushes': pushes,
        'hit_rate': None if decisions == 0 else round(wins / decisions, 4),
        'profit_units': round(profit, 4),
        'roi_per_bet': None if n == 0 else round(profit / n, 4),
    }


def main():
    ledger = load_ledger()
    settled = ledger[ledger['result_status'].astype(str).isin(['win','loss','push'])].copy()
    report = {
        'method': 'forward-only sportsbook audit; no reconstructed historical ROI',
        'overall': _summary(settled),
        'by_market': {},
        'by_model_version': {},
        'pending': int((ledger['result_status'].astype(str) == 'pending').sum()) if not ledger.empty else 0,
    }
    if not settled.empty:
        for market, grp in settled.groupby('market', dropna=False):
            report['by_market'][str(market)] = _summary(grp)
        for version, grp in settled.groupby('model_version', dropna=False):
            report['by_model_version'][str(version)] = _summary(grp)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
