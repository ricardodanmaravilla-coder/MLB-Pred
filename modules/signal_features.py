"""Leak-safe advanced signals for MLB-Pred.

Only information that can be aligned to a prior completed season is admitted here.
Live-only signals (weather, confirmed starter, current bullpen) stay in Monte Carlo
unless an equivalent historical pregame dataset exists.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .team_utils import normalize_team

BATTING_SIGNAL_COLUMNS = ('wOBA','ISO','BB%','K%')
PITCHING_SIGNAL_COLUMNS = ('FIP','xFIP','WHIP','K-BB%','GB%','HR/9')
ADVANCED_SIGNAL_COLUMNS = [
    'home_rest_norm','away_rest_norm',
    'home_woba_rel','away_woba_rel','home_iso_rel','away_iso_rel',
    'home_bb_rel','away_bb_rel','home_k_rel','away_k_rel',
    'home_fip_rel','away_fip_rel','home_xfip_rel','away_xfip_rel',
    'home_whip_rel','away_whip_rel','home_kbb_rel','away_kbb_rel',
    'home_gb_rel','away_gb_rel','home_hr9_rel','away_hr9_rel',
]

def _num_series(s): return pd.to_numeric(s, errors='coerce')

def _season_team_maps(df, columns):
    if df is None or df.empty or 'Team' not in df.columns or 'Season' not in df.columns: return {}, {}
    x=df.copy(); x['Team']=x['Team'].map(normalize_team); x['Season']=_num_series(x['Season'])
    maps={}; medians={}
    for c in columns:
        if c not in x.columns: continue
        x[c]=_num_series(x[c]); valid=x.dropna(subset=['Team','Season',c])
        maps[c]=valid.set_index(['Team','Season'])[c].to_dict(); medians[c]=valid.groupby('Season')[c].median().to_dict()
    return maps, medians

def _relative(value, center, inverse=False):
    try:
        v=float(value); c=float(center)
        if not np.isfinite(v) or not np.isfinite(c) or abs(c)<1e-9: return 1.0
        ratio=(c/v) if inverse else (v/c)
        return float(np.clip(ratio,0.65,1.35))
    except Exception: return 1.0

def build_advanced_signal_frame(feature_frame, batting, pitching):
    if feature_frame is None or feature_frame.empty: return pd.DataFrame(columns=ADVANCED_SIGNAL_COLUMNS)
    bm,bmed=_season_team_maps(batting,BATTING_SIGNAL_COLUMNS); pm,pmed=_season_team_maps(pitching,PITCHING_SIGNAL_COLUMNS)
    rows=[]
    for r in feature_frame.itertuples(index=False):
        sy=int(r.Season)-1; h=normalize_team(r.Home); a=normalize_team(r.Away)
        d={'home_rest_norm':float(np.clip(float(getattr(r,'home_rest_days',3.0))/3.0,0.0,2.0)), 'away_rest_norm':float(np.clip(float(getattr(r,'away_rest_days',3.0))/3.0,0.0,2.0))}
        for col,key in {'wOBA':'woba','ISO':'iso','BB%':'bb','K%':'k'}.items():
            center=bmed.get(col,{}).get(sy); hm=bm.get(col,{}).get((h,sy),center); am=bm.get(col,{}).get((a,sy),center); inv=(col=='K%')
            d[f'home_{key}_rel']=_relative(hm,center,inv); d[f'away_{key}_rel']=_relative(am,center,inv)
        for col,key in {'FIP':'fip','xFIP':'xfip','WHIP':'whip','K-BB%':'kbb','GB%':'gb','HR/9':'hr9'}.items():
            center=pmed.get(col,{}).get(sy); hm=pm.get(col,{}).get((h,sy),center); am=pm.get(col,{}).get((a,sy),center); inv=col in ('FIP','xFIP','WHIP','HR/9')
            d[f'home_{key}_rel']=_relative(hm,center,inv); d[f'away_{key}_rel']=_relative(am,center,inv)
        rows.append(d)
    out=pd.DataFrame(rows,index=feature_frame.index)
    for c in ADVANCED_SIGNAL_COLUMNS:
        if c not in out.columns: out[c]=1.0
    return out[ADVANCED_SIGNAL_COLUMNS].replace([np.inf,-np.inf],np.nan).fillna(1.0)

def coverage_report(batting,pitching):
    report={}
    for name,df,cols in [('batting',batting,BATTING_SIGNAL_COLUMNS),('pitching',pitching,PITCHING_SIGNAL_COLUMNS)]:
        for c in cols:
            report[f'{name}:{c}']=0.0 if df is None or df.empty or c not in df.columns else round(float(_num_series(df[c]).notna().mean()),4)
    return report
