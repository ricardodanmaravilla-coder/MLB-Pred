"""Metric-quality helpers shared by training and live prediction."""
from __future__ import annotations
import pandas as pd

REAL_COVERAGE_MIN=0.80

def _source_coverage(df,metric_col,source_col,marker='FANGRAPHS_REAL'):
    if df is None or df.empty or metric_col not in df.columns or source_col not in df.columns:return 0.0
    usable=pd.to_numeric(df[metric_col],errors='coerce').notna();n=int(usable.sum())
    if not n:return 0.0
    return float(df.loc[usable,source_col].astype(str).str.contains(marker,na=False).mean())

def batting_metric(df):
    if df is None or df.empty:return None
    if 'wRC+' in df.columns and _source_coverage(df,'wRC+','wRC+_Source')>=REAL_COVERAGE_MIN:return 'wRC+'
    if 'OPS_Index' in df.columns and pd.to_numeric(df['OPS_Index'],errors='coerce').notna().mean()>=REAL_COVERAGE_MIN:return 'OPS_Index'
    if 'wRC+' in df.columns:
        vals=pd.to_numeric(df['wRC+'],errors='coerce')
        if vals.notna().mean()>=REAL_COVERAGE_MIN and vals.between(50,150).mean()>0.8 and vals.median()>85:return 'wRC+'
    return None

def pitching_metric(df):
    if df is None or df.empty:return None
    if 'xFIP' in df.columns and _source_coverage(df,'xFIP','xFIP_Source')>=REAL_COVERAGE_MIN:return 'xFIP'
    if 'FIP' in df.columns and pd.to_numeric(df['FIP'],errors='coerce').notna().mean()>=REAL_COVERAGE_MIN:return 'FIP'
    if 'SIERA' in df.columns and pd.to_numeric(df['SIERA'],errors='coerce').notna().mean()>=REAL_COVERAGE_MIN:return 'SIERA'
    if 'ERA' in df.columns and pd.to_numeric(df['ERA'],errors='coerce').notna().mean()>=REAL_COVERAGE_MIN:return 'ERA'
    if 'xFIP' in df.columns:
        vals=pd.to_numeric(df['xFIP'],errors='coerce')
        if vals.notna().mean()>=REAL_COVERAGE_MIN:return 'xFIP'
    return None

def row_pitching_value(row,fallback=None):
    """Prefer real xFIP/FIP/SIERA for one pitcher or bullpen row, then ERA."""
    try:
        source=str(row.get('xFIP_Source',row.get('Source','')))
        if 'FANGRAPHS_REAL' in source:
            for c in ('xFIP','FIP','SIERA','ERA'):
                v=pd.to_numeric(pd.Series([row.get(c)]),errors='coerce').iloc[0]
                if pd.notna(v):return float(v),c
        for c in ('FIP','xFIP','SIERA'):
            v=pd.to_numeric(pd.Series([row.get(c)]),errors='coerce').iloc[0]
            if pd.notna(v) and 'LEGACY' not in source:return float(v),c
        era=pd.to_numeric(pd.Series([row.get('ERA')]),errors='coerce').iloc[0]
        if pd.notna(era):return float(era),'ERA'
    except Exception:pass
    return fallback,'fallback'
