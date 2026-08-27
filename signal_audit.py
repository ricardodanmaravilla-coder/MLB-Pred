"""Inspect advanced-signal coverage without mutating production data."""
import json
import pandas as pd
from modules.signal_features import coverage_report


def main():
    bat=pd.read_csv('data/mlb_batting.csv'); pit=pd.read_csv('data/mlb_pitching.csv')
    report=coverage_report(bat,pit)
    eligible={k:v for k,v in report.items() if v>=0.65}
    print(json.dumps({'coverage':report,'eligible_ge_65pct':eligible,'eligible_count':len(eligible)},indent=2,sort_keys=True))

if __name__=='__main__': main()
