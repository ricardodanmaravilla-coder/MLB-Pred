"""Metric-quality helpers shared by training and live prediction."""
from __future__ import annotations

import pandas as pd


def batting_metric(df):
    if df is None or df.empty:
        return None
    if 'wRC+' in df.columns:
        if 'wRC+_Source' in df.columns:
            src = df['wRC+_Source'].astype(str)
            if src.str.contains('FANGRAPHS_REAL', na=False).any():
                return 'wRC+'
        vals = pd.to_numeric(df['wRC+'], errors='coerce')
        if vals.between(50, 150).mean() > 0.8 and vals.median() > 85:
            return 'wRC+'
    if 'OPS_Index' in df.columns:
        return 'OPS_Index'
    if 'wRC+' in df.columns:
        return 'wRC+'
    return None


def pitching_metric(df):
    if df is None or df.empty:
        return None
    if 'xFIP' in df.columns:
        if 'xFIP_Source' in df.columns:
            src = df['xFIP_Source'].astype(str)
            if src.str.contains('FANGRAPHS_REAL', na=False).any():
                return 'xFIP'
        vals = pd.to_numeric(df['xFIP'], errors='coerce')
        era = pd.to_numeric(df.get('ERA'), errors='coerce') if 'ERA' in df.columns else None
        if era is not None and vals.notna().sum() and (vals - era).abs().fillna(0).gt(0.01).mean() > 0.2:
            return 'xFIP'
    if 'FIP' in df.columns and pd.to_numeric(df['FIP'], errors='coerce').notna().mean() > 0.7:
        return 'FIP'
    if 'ERA' in df.columns:
        return 'ERA'
    if 'xFIP' in df.columns:
        return 'xFIP'
    return None


def row_pitching_value(row, fallback=None):
    """Prefer real xFIP/FIP for one pitcher or bullpen row, then ERA."""
    try:
        source = str(row.get('xFIP_Source', row.get('Source', '')))
        if 'FANGRAPHS_REAL' in source:
            for c in ('xFIP', 'FIP', 'ERA'):
                v = pd.to_numeric(pd.Series([row.get(c)]), errors='coerce').iloc[0]
                if pd.notna(v):
                    return float(v), c
        for c in ('xFIP', 'FIP'):
            v = pd.to_numeric(pd.Series([row.get(c)]), errors='coerce').iloc[0]
            if pd.notna(v) and c == 'xFIP' and 'LEGACY' not in source:
                return float(v), c
        v = pd.to_numeric(pd.Series([row.get('ERA')]), errors='coerce').iloc[0]
        if pd.notna(v):
            return float(v), 'ERA'
    except Exception:
        pass
    return fallback, 'fallback'
