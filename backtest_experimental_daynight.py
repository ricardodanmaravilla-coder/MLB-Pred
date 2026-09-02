"""Run the existing strict temporal gate with starter Day/Night research features.

This wrapper changes only the experimental candidate feature registry. It reuses
exactly the same models, 70/15/15 chronology, selection rules and final holdout
promotion gate from backtest_experimental_parquet.py.
"""
import backtest_experimental_parquet as bt

METRICS = ("era","whip","k_pct","bb_pct","kbb_pct","hr9")
DAYNIGHT_FEATURES = []
for side in ("home","away"):
    for metric in METRICS:
        DAYNIGHT_FEATURES.extend([f"{side}_starter_dn_{metric}", f"{side}_starter_dn_delta_{metric}"])
    DAYNIGHT_FEATURES.extend([f"{side}_starter_dn_ip", f"{side}_starter_dn_weight"])

# Make the new post-Parquet columns eligible for the existing >=50% coverage gate.
bt.RESEARCH_FEATURES = list(dict.fromkeys(list(bt.RESEARCH_FEATURES) + DAYNIGHT_FEATURES))

# Evaluate the baseball concept coherently rather than cherry-picking individual
# split statistics. Keep the existing groups and gate unchanged.
bt.SEMANTIC_GROUPS = list(bt.SEMANTIC_GROUPS)
insert_at = next((i for i,(name,_) in enumerate(bt.SEMANTIC_GROUPS) if name == "bullpen_workload"), 1)
bt.SEMANTIC_GROUPS.insert(insert_at, ("starter_day_night", DAYNIGHT_FEATURES))

if __name__ == "__main__":
    bt.run_backtest()
