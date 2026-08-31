"""Pre-specified confirmed-lineup + real starter-hand validation.

Validation-only experiment.  Extends the existing leak-safe lineup walk-forward
features with the actual opposing starter hand from the official game feed and
historical hitter platoon splits when available.  Never substitutes future
results or fabricated split values.

This first strict version intentionally delegates the stable chronological
baseline/lineup construction to backtest_lineup_walkforward.py, then requires
its output to expose pregame platoon columns.  If those columns are absent it
fails loudly rather than silently pretending a platoon test occurred.
"""
from __future__ import annotations
import subprocess, sys
from pathlib import Path
import pandas as pd

BASE=Path('artifacts/lineup_walkforward_predictions.csv')
OUT=Path('artifacts/lineup_platoon_walkforward_predictions.csv')
REQUIRED={'Date','actual_home_win','baseline_home_prob','candidate_home_prob'}
PLATOON_CANDIDATES={'lu_ops_vs_hand_diff','lu_obp_vs_hand_diff','lu_slg_vs_hand_diff','home_starter_hand','away_starter_hand'}


def main():
    subprocess.run([sys.executable,'backtest_lineup_walkforward.py'],check=True)
    d=pd.read_csv(BASE)
    missing=REQUIRED-set(d.columns)
    if missing: raise RuntimeError(f'Base lineup output missing required columns: {sorted(missing)}')
    present=PLATOON_CANDIDATES & set(d.columns)
    if not present:
        raise RuntimeError('Strict platoon validation blocked: existing historical pipeline does not yet expose pregame hitter-vs-starter-hand split features. No proxy will be fabricated. Build official historical starter hand + pregame hitter split history before promotion testing.')
    # When the upstream historical builder exposes the verified platoon fields,
    # preserve the paired chronological predictions for the strict date-block gate.
    d.to_csv(OUT,index=False)
    print(f'OK {OUT} rows={len(d)} platoon_columns={sorted(present)}')

if __name__=='__main__': main()
