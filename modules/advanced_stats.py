"""Optional advanced MLB statistics enrichment.

Fail-soft by design: official MLB StatsAPI remains the durable baseline. FanGraphs
metrics are overlaid only when available and every real metric keeps provenance.
"""
from __future__ import annotations

import math
from typing import Iterable
import pandas as pd

from .team_utils import normalize_team

BATTING_ADVANCED = (
    'wRC+','wOBA','ISO','BB%','K%','BABIP','WAR','Off','BsR',
    'EV','maxEV','HardHit%','Barrel%','Pull%','Cent%','Oppo%','GB%','FB%','LD%'
)
PITCHING_ADVANCED = (
    'ERA','FIP','xFIP','SIERA','WHIP','K-BB%','K%','BB%','GB%','HR/9','WAR','IP'
)


def _num(v):
    try:
        x=float(v); return x if math.isfinite(x) else None
    except (TypeError,ValueError): return None


def _normalized_team_column(df):
    if 'Team' not in df.columns: return pd.Series([None]*len(df),index=df.index)
    return df['Team'].map(normalize_team)


def _copy_metric(src,dst,source_col,target_col):
    if source_col in src.columns: dst[target_col]=pd.to_numeric(src[source_col],errors='coerce')


def fetch_fangraphs_team_season(season:int):
    from pybaseball import team_batting,team_pitching,team_pitching_relievers
    bat=team_batting(int(season),int(season),ind=1); pit=team_pitching(int(season),int(season),ind=1); rel=team_pitching_relievers(int(season),int(season),ind=1)
    for frame in (bat,pit,rel):
        if frame is not None and not frame.empty:
            frame['Team']=_normalized_team_column(frame); frame['Season']=int(season)
    return bat,pit,rel


def _overlay(base,adv,season,metrics,source_metric,source_col,source_tag):
    if adv is None or adv.empty: return base
    keep=pd.DataFrame({'Team':_normalized_team_column(adv),'Season':season})
    for c in metrics: _copy_metric(adv,keep,c,c)
    keep=keep.dropna(subset=['Team']).drop_duplicates(['Team','Season'])
    if source_metric in keep.columns:
        keep[source_col]=None
        keep.loc[pd.to_numeric(keep[source_metric],errors='coerce').notna(),source_col]=source_tag
    out=base.merge(keep,on=['Team','Season'],how='left',suffixes=('','_FG'))
    for c in tuple(metrics)+(source_col,):
        fg=f'{c}_FG'
        if fg in out.columns:
            existing=out[c] if c in out.columns else pd.Series(index=out.index,dtype='object')
            out[c]=out[fg].combine_first(existing); out.drop(columns=[fg],inplace=True)
    return out


def enrich_team_frames(base_batting:pd.DataFrame,base_pitching:pd.DataFrame,seasons:Iterable[int]):
    bat=base_batting.copy(); pit=base_pitching.copy(); bullpen_rows=[]
    if bat.empty or pit.empty: return bat,pit,pd.DataFrame()
    bat['Team']=bat['Team'].map(normalize_team); pit['Team']=pit['Team'].map(normalize_team)
    bat['Season']=pd.to_numeric(bat['Season'],errors='coerce'); pit['Season']=pd.to_numeric(pit['Season'],errors='coerce')
    for season in sorted({int(s) for s in seasons}):
        try: fg_bat,fg_pit,fg_rel=fetch_fangraphs_team_season(season)
        except Exception as exc:
            print(f'⚠️ FanGraphs {season} no disponible: {exc}'); continue
        bat=_overlay(bat,fg_bat,season,BATTING_ADVANCED,'wRC+','wRC+_Source','FANGRAPHS_REAL_WRCPLUS')
        pit=_overlay(pit,fg_pit,season,PITCHING_ADVANCED,'xFIP','xFIP_Source','FANGRAPHS_REAL_XFIP')
        if fg_rel is not None and not fg_rel.empty:
            for _,r in fg_rel.iterrows():
                team=normalize_team(r.get('Team')); era=_num(r.get('ERA'))
                if not team or era is None: continue
                row={'Team':team,'Season':season,'ERA':era,'Source':'FANGRAPHS_REAL_RELIEVERS'}
                for c in PITCHING_ADVANCED:
                    if c!='ERA': row[c]=_num(r.get(c))
                bullpen_rows.append(row)
    bullpen=pd.DataFrame(bullpen_rows)
    if not bullpen.empty: bullpen=bullpen.drop_duplicates(['Team','Season'],keep='last').sort_values(['Season','Team'])
    return bat,pit,bullpen


def enrich_pitcher_frame(base:pd.DataFrame,season:int):
    if base is None or base.empty: return base
    try:
        from pybaseball import pitching_stats
        fg=pitching_stats(int(season),int(season),qual=0,ind=1)
    except Exception as exc:
        print(f'⚠️ FanGraphs pitchers no disponibles: {exc}'); return base
    if fg is None or fg.empty or 'Name' not in fg.columns: return base
    cols=['Name']+[c for c in ('Team',)+PITCHING_ADVANCED+('GS','G') if c in fg.columns]
    adv=fg[cols].copy()
    if 'Team' in adv.columns: adv['Team']=adv['Team'].map(normalize_team)
    for c in cols:
        if c not in ('Name','Team'): adv[c]=pd.to_numeric(adv[c],errors='coerce')
    adv=adv.drop_duplicates(['Name','Team'] if 'Team' in adv.columns else ['Name'],keep='last')
    out=base.copy(); out['Team']=out['Team'].map(normalize_team)
    keys=['Name','Team'] if 'Team' in adv.columns else ['Name']; out=out.merge(adv,on=keys,how='left',suffixes=('','_FG'))
    real_xfip=pd.to_numeric(out['xFIP_FG'],errors='coerce').notna() if 'xFIP_FG' in out.columns else pd.Series(False,index=out.index)
    for c in PITCHING_ADVANCED+('GS','G'):
        fgcol=f'{c}_FG'
        if fgcol in out.columns:
            out[c]=out[fgcol].combine_first(out[c] if c in out.columns else pd.Series(index=out.index,dtype=float)); out.drop(columns=[fgcol],inplace=True)
    if 'xFIP_Source' not in out.columns: out['xFIP_Source']=None
    out.loc[real_xfip,'xFIP_Source']='FANGRAPHS_REAL_XFIP'
    return out
