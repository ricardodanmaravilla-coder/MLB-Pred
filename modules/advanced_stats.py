"""Optional advanced MLB statistics enrichment.

This module is deliberately fail-soft: production keeps the official MLB StatsAPI
baseline when FanGraphs/pybaseball is unavailable. When available, real FanGraphs
wRC+, FIP/xFIP and reliever metrics replace legacy proxies and are explicitly tagged.
"""
from __future__ import annotations

import math
from typing import Iterable

import pandas as pd

from .team_utils import normalize_team


def _num(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _normalized_team_column(df: pd.DataFrame) -> pd.Series:
    if 'Team' not in df.columns:
        return pd.Series([None] * len(df), index=df.index)
    return df['Team'].map(normalize_team)


def _copy_metric(src, dst, source_col, target_col):
    if source_col in src.columns:
        dst[target_col] = pd.to_numeric(src[source_col], errors='coerce')


def fetch_fangraphs_team_season(season: int):
    """Return batting, all-pitching and reliever team tables for one season."""
    from pybaseball import team_batting, team_pitching, team_pitching_relievers

    bat = team_batting(int(season), int(season), ind=1)
    pit = team_pitching(int(season), int(season), ind=1)
    rel = team_pitching_relievers(int(season), int(season), ind=1)
    for frame in (bat, pit, rel):
        if frame is not None and not frame.empty:
            frame['Team'] = _normalized_team_column(frame)
            frame['Season'] = int(season)
    return bat, pit, rel


def enrich_team_frames(base_batting: pd.DataFrame, base_pitching: pd.DataFrame,
                       seasons: Iterable[int]):
    """Overlay real FanGraphs metrics onto StatsAPI team-season baselines."""
    bat = base_batting.copy()
    pit = base_pitching.copy()
    bullpen_rows = []
    if bat.empty or pit.empty:
        return bat, pit, pd.DataFrame()

    bat['Team'] = bat['Team'].map(normalize_team)
    pit['Team'] = pit['Team'].map(normalize_team)
    bat['Season'] = pd.to_numeric(bat['Season'], errors='coerce')
    pit['Season'] = pd.to_numeric(pit['Season'], errors='coerce')

    for season in sorted({int(s) for s in seasons}):
        try:
            fg_bat, fg_pit, fg_rel = fetch_fangraphs_team_season(season)
        except Exception as exc:
            print(f'⚠️ FanGraphs {season} no disponible: {exc}')
            continue

        if fg_bat is not None and not fg_bat.empty:
            keep = pd.DataFrame({'Team': _normalized_team_column(fg_bat), 'Season': season})
            for c in ('wRC+', 'wOBA', 'ISO', 'BB%', 'K%', 'BABIP', 'WAR'):
                _copy_metric(fg_bat, keep, c, c)
            keep = keep.dropna(subset=['Team']).drop_duplicates(['Team','Season'])
            if 'wRC+' in keep.columns:
                keep['wRC+_Source'] = None
                keep.loc[pd.to_numeric(keep['wRC+'], errors='coerce').notna(), 'wRC+_Source'] = 'FANGRAPHS_REAL_WRCPLUS'
            bat = bat.merge(keep, on=['Team','Season'], how='left', suffixes=('', '_FG'))
            for c in ('wRC+','wOBA','ISO','BB%','K%','BABIP','WAR','wRC+_Source'):
                fg = f'{c}_FG'
                if fg in bat.columns:
                    bat[c] = bat[fg].combine_first(bat[c] if c in bat.columns else pd.Series(index=bat.index, dtype='object'))
                    bat.drop(columns=[fg], inplace=True)

        if fg_pit is not None and not fg_pit.empty:
            keep = pd.DataFrame({'Team': _normalized_team_column(fg_pit), 'Season': season})
            for c in ('ERA','FIP','xFIP','WHIP','K-BB%','GB%','HR/9','WAR'):
                _copy_metric(fg_pit, keep, c, c)
            keep = keep.dropna(subset=['Team']).drop_duplicates(['Team','Season'])
            if 'xFIP' in keep.columns:
                keep['xFIP_Source'] = None
                keep.loc[pd.to_numeric(keep['xFIP'], errors='coerce').notna(), 'xFIP_Source'] = 'FANGRAPHS_REAL_XFIP'
            pit = pit.merge(keep, on=['Team','Season'], how='left', suffixes=('', '_FG'))
            for c in ('ERA','FIP','xFIP','WHIP','K-BB%','GB%','HR/9','WAR','xFIP_Source'):
                fg = f'{c}_FG'
                if fg in pit.columns:
                    pit[c] = pit[fg].combine_first(pit[c] if c in pit.columns else pd.Series(index=pit.index, dtype='object'))
                    pit.drop(columns=[fg], inplace=True)

        if fg_rel is not None and not fg_rel.empty:
            for _, r in fg_rel.iterrows():
                team = normalize_team(r.get('Team'))
                era, xfip, fip = _num(r.get('ERA')), _num(r.get('xFIP')), _num(r.get('FIP'))
                ip = _num(r.get('IP'))
                if not team or era is None:
                    continue
                bullpen_rows.append({
                    'Team': team, 'Season': season, 'ERA': era,
                    'xFIP': xfip, 'FIP': fip, 'IP': ip,
                    'K-BB%': _num(r.get('K-BB%')), 'WHIP': _num(r.get('WHIP')),
                    'GB%': _num(r.get('GB%')), 'HR/9': _num(r.get('HR/9')),
                    'Source': 'FANGRAPHS_REAL_RELIEVERS',
                })

    bullpen = pd.DataFrame(bullpen_rows)
    if not bullpen.empty:
        bullpen = bullpen.drop_duplicates(['Team','Season'], keep='last').sort_values(['Season','Team'])
    return bat, pit, bullpen


def enrich_pitcher_frame(base: pd.DataFrame, season: int):
    """Overlay real FanGraphs FIP/xFIP/K-BB% onto current individual pitchers.

    A row is tagged FANGRAPHS_REAL_XFIP only when the FanGraphs-side xFIP value
    itself matched. Legacy ERA-derived xFIP fallbacks remain explicitly legacy.
    """
    if base is None or base.empty:
        return base
    try:
        from pybaseball import pitching_stats
        fg = pitching_stats(int(season), int(season), qual=0, ind=1)
    except Exception as exc:
        print(f'⚠️ FanGraphs pitchers no disponibles: {exc}')
        return base
    if fg is None or fg.empty or 'Name' not in fg.columns:
        return base

    cols = ['Name'] + [c for c in ('Team','ERA','FIP','xFIP','WHIP','K-BB%','GB%','HR/9','IP','GS','G') if c in fg.columns]
    adv = fg[cols].copy()
    if 'Team' in adv.columns:
        adv['Team'] = adv['Team'].map(normalize_team)
    for c in cols:
        if c not in ('Name','Team'):
            adv[c] = pd.to_numeric(adv[c], errors='coerce')
    adv = adv.drop_duplicates(['Name','Team'] if 'Team' in adv.columns else ['Name'], keep='last')

    out = base.copy()
    out['Team'] = out['Team'].map(normalize_team)
    keys = ['Name','Team'] if 'Team' in adv.columns else ['Name']
    out = out.merge(adv, on=keys, how='left', suffixes=('', '_FG'))

    # Capture the provenance mask before combine_first can fill missing FG values
    # from legacy StatsAPI columns.
    real_xfip = (
        pd.to_numeric(out['xFIP_FG'], errors='coerce').notna()
        if 'xFIP_FG' in out.columns else pd.Series(False, index=out.index)
    )

    for c in ('ERA','FIP','xFIP','WHIP','K-BB%','GB%','HR/9','IP','GS','G'):
        fgcol = f'{c}_FG'
        if fgcol in out.columns:
            out[c] = out[fgcol].combine_first(out[c] if c in out.columns else pd.Series(index=out.index, dtype=float))
            out.drop(columns=[fgcol], inplace=True)

    if 'xFIP_Source' not in out.columns:
        out['xFIP_Source'] = None
    out.loc[real_xfip, 'xFIP_Source'] = 'FANGRAPHS_REAL_XFIP'
    return out
