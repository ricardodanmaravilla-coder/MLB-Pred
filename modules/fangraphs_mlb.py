"""Optional FanGraphs enrichment via pybaseball.

Team batting/pitching are fetched independently so an unavailable reliever/player
endpoint cannot disable the working team-level advanced metrics. Individual pitcher
scrapes are best-effort only because FanGraphs can return 403 from legacy endpoints.
"""
from __future__ import annotations
import pandas as pd
from .team_utils import normalize_team


def _num(df,col):
    if col in df.columns: df[col]=pd.to_numeric(df[col],errors='coerce')
def _team_key(value):
    key=normalize_team(value); return str(key).upper() if key else None

def _clean_team(df,kind):
    if df is None or len(df)==0: return pd.DataFrame()
    df=df.copy(); team_col=next((c for c in ('Team','team','Name') if c in df.columns),None); season_col=next((c for c in ('Season','season','Year') if c in df.columns),None)
    if team_col is None or season_col is None: return pd.DataFrame()
    df['Team']=df[team_col].map(_team_key); df['Season']=pd.to_numeric(df[season_col],errors='coerce'); df=df.dropna(subset=['Team','Season']); df['Season']=df['Season'].astype(int)
    for c in ('wRC+','OPS','wOBA','ISO','BB%','K%','K-BB%','FIP','xFIP','ERA','WHIP','HR/9','GB%','IP'):_num(df,c)
    df['DataSource']=f'FANGRAPHS_{kind}'; return df

def fetch_team_fangraphs(start_season,end_season=None):
    end_season=int(end_season or start_season); bat=pd.DataFrame(); pit=pd.DataFrame(); rel=pd.DataFrame()
    try:
        from pybaseball import team_batting
        bat=_clean_team(team_batting(int(start_season),end_season,ind=1),'TEAM_BATTING')
    except Exception as exc: print(f'⚠️ FanGraphs team batting unavailable: {exc}')
    try:
        from pybaseball import team_pitching
        pit=_clean_team(team_pitching(int(start_season),end_season,ind=1),'TEAM_PITCHING')
    except Exception as exc: print(f'⚠️ FanGraphs team pitching unavailable: {exc}')
    # pybaseball 2.2.x does not reliably expose a reliever-team helper. Keep this
    # optional and never let it invalidate batting/pitching already retrieved.
    try:
        import pybaseball
        fn=getattr(pybaseball,'team_pitching_relievers',None)
        if callable(fn): rel=_clean_team(fn(int(start_season),end_season,ind=1),'RELIEVERS')
    except Exception as exc: print(f'⚠️ FanGraphs reliever aggregate unavailable: {exc}')
    return bat,pit,rel

def fetch_pitcher_fangraphs(season):
    try:
        from pybaseball import pitching_stats
        df=pitching_stats(int(season),int(season),qual=0,ind=1)
        if df is None or len(df)==0:return pd.DataFrame()
        df=df.copy(); name_col=next((c for c in ('Name','name','Player') if c in df.columns),None); team_col=next((c for c in ('Team','team') if c in df.columns),None)
        if name_col is None or team_col is None:return pd.DataFrame()
        df['Name']=df[name_col].astype(str); df['Team']=df[team_col].map(_team_key); df['Season']=int(season)
        for c in ('ERA','FIP','xFIP','WHIP','K/9','BB/9','HR/9','K%','BB%','K-BB%','GB%','IP','GS','G'):_num(df,c)
        df['DataSource']='FANGRAPHS_PITCHER'; return df.dropna(subset=['Name','Team'])
    except Exception as exc:
        print(f'⚠️ FanGraphs pitcher enrichment unavailable: {exc}'); return pd.DataFrame()

def prefer_real_metric(df,primary,fallback):
    if df is None or df.empty:return pd.Series(dtype=float)
    a=pd.to_numeric(df[primary],errors='coerce') if primary in df.columns else pd.Series(index=df.index,dtype=float); b=pd.to_numeric(df[fallback],errors='coerce') if fallback in df.columns else pd.Series(index=df.index,dtype=float); return a.combine_first(b)
