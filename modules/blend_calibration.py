"""Forward-only calibration of ML vs Monte Carlo blend weights.

Historical sportsbook/context data are incomplete, so weights are learned only from
settled live snapshots. Small samples stay at 50/50. With enough decisions, a Brier-
optimal weight is estimated per market and shrunk strongly toward 0.50 to reduce
overfitting/selection noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .pick_ledger import load_ledger

DEFAULT_WEIGHT = 0.50
MIN_DECISIONS = 80
MAX_DEVIATION = 0.20


def _numeric_pct(series):
    return pd.to_numeric(series.astype(str).str.replace('%','',regex=False), errors='coerce') / 100.0


def learn_blend_weight(market, ledger=None, min_decisions=MIN_DECISIONS):
    try:
        df = load_ledger() if ledger is None else ledger.copy()
        if df is None or df.empty: return DEFAULT_WEIGHT, 0, 'default_no_data'
        x = df[df['market'].astype(str) == str(market)].copy()
        x = x[x['result_status'].isin(['win','loss'])]
        x['pml'] = _numeric_pct(x['prob_ml']); x['pmc'] = _numeric_pct(x['prob_mc'])
        x = x.dropna(subset=['pml','pmc'])
        n = len(x)
        if n < int(min_decisions): return DEFAULT_WEIGHT, n, 'default_small_sample'
        y = (x['result_status'] == 'win').astype(float).to_numpy()
        pml = x['pml'].clip(.01,.99).to_numpy(); pmc = x['pmc'].clip(.01,.99).to_numpy()
        grid = np.linspace(0.20,0.80,61)
        scores = [float(np.mean(((w*pml + (1-w)*pmc) - y)**2)) for w in grid]
        raw = float(grid[int(np.argmin(scores))])
        # Empirical-Bayes style shrinkage: even at 80 bets only 50% of the raw move
        # from equal weighting is trusted; confidence rises gradually with sample size.
        reliability = min(0.90, n / (n + 80.0))
        weight = 0.50 + (raw - 0.50) * reliability
        weight = float(np.clip(weight, 0.50-MAX_DEVIATION, 0.50+MAX_DEVIATION))
        return round(weight,3), n, 'forward_brier_shrunk'
    except Exception:
        return DEFAULT_WEIGHT, 0, 'default_error'


def market_blend_weights(ledger=None):
    out = {}
    for market in ('Moneyline','Totales','Hándicap'):
        w,n,source = learn_blend_weight(market, ledger=ledger)
        out[market] = {'ml_weight':w, 'mc_weight':round(1-w,3), 'n':n, 'source':source}
    return out
