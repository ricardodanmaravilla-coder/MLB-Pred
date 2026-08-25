"""Optional FanGraphs enrichment via pybaseball.

The production miners must never fail just because FanGraphs is unavailable. These
helpers return empty frames on network/schema failures so MLB StatsAPI remains the
safe fallback. When available, they replace legacy OPS*100 / ERA proxies with real
FanGraphs wRC+, FIP/xFIP and related rate metrics.
"""

from __future__ import annotations

import pandas as pd

from .team_utils import normalize_team


def _num(df, col):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')


def _team_key(value):
    key = normalize_team(value)
    return str(key).upper() if key else None


def fetch_team_fangraphs(start_season, end_season=None):
    try:
        from pybaseball import team_batting, team_pitching, team_pitching_relievers

        end_season = int(end_season or start_season)
        bat = team_batting(int(start_season), end_season, ind=1)
        pit = team_pitching(int(start_season), end_season, ind=1)
        rel = team_pitching_relievers(int(start_season), end_season, ind=1)

        def clean(df, kind):
            if df is None or len(df) == 0:
                return pd.DataFrame()
            df = df.copy()
            team_col = next((c for c in ('Team', 'team', 'Name') if c in df.columns), None)
            season_col = next((c for c in ('Season', 'season', 'Year') if c in df.columns), None)
            if team_col is None or season_col is None:
                return pd.DataFrame()
            df['Team'] = df[team_col].map(_team_key)
            df['Season'] = pd.to_numeric(df[season_col], errors='coerce')
            df = df.dropna(subset=['Team', 'Season'])
            df['Season'] = df['Season'].astype(int)
            for c in ('wRC+', 'OPS', 'wOBA', 'ISO', 'BB%', 'K%', 'K-BB%', 'FIP', 'xFIP', 'ERA', 'WHIP', 'HR/9', 'GB%'):
                _num(df, c)
            df['DataSource'] = f'FANGRAPHS_{kind}'
            return df

        return clean(bat, 'TEAM_BATTING'), clean(pit, 'TEAM_PITCHING'), clean(rel, 'RELIEVERS')
    except Exception as exc:
        print(f'⚠️ FanGraphs team enrichment unavailable: {exc}')
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def fetch_pitcher_fangraphs(season):
    try:
        from pybaseball import pitching_stats

        df = pitching_stats(int(season), int(season), qual=0, ind=1)
        if df is None or len(df) == 0:
            return pd.DataFrame()
        df = df.copy()
        name_col = next((c for c in ('Name', 'name', 'Player') if c in df.columns), None)
        team_col = next((c for c in ('Team', 'team') if c in df.columns), None)
        if name_col is None or team_col is None:
            return pd.DataFrame()
        df['Name'] = df[name_col].astype(str)
        df['Team'] = df[team_col].map(_team_key)
        df['Season'] = int(season)
        for c in ('ERA', 'FIP', 'xFIP', 'WHIP', 'K/9', 'BB/9', 'HR/9', 'K%', 'BB%', 'K-BB%', 'GB%', 'IP', 'GS', 'G'):
            _num(df, c)
        df['DataSource'] = 'FANGRAPHS_PITCHER'
        return df.dropna(subset=['Name', 'Team'])
    except Exception as exc:
        print(f'⚠️ FanGraphs pitcher enrichment unavailable: {exc}')
        return pd.DataFrame()


def prefer_real_metric(df, primary, fallback):
    """Return a numeric Series preferring a real advanced metric when present."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    a = pd.to_numeric(df[primary], errors='coerce') if primary in df.columns else pd.Series(index=df.index, dtype=float)
    b = pd.to_numeric(df[fallback], errors='coerce') if fallback in df.columns else pd.Series(index=df.index, dtype=float)
    return a.combine_first(b)
