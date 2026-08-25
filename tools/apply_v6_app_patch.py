from pathlib import Path

p=Path('app_mlb.py')
t=p.read_text(encoding='utf-8')

def rep(old,new,label,count=None):
    global t
    n=t.count(old)
    if n==0 or (count is not None and n!=count):
        raise SystemExit(f'{label}: found {n}')
    t=t.replace(old,new)

rep("from modules.pick_ledger import append_snapshot, persistent_backend_available\n", "from modules.pick_ledger import append_snapshot, persistent_backend_available\nfrom modules.metric_quality import batting_metric, pitching_metric, row_pitching_value\n", 'metric imports',1)

old='''        col = 'OPS_Index' if 'OPS_Index' in df.columns else ('wRC+' if 'wRC+' in df.columns else None)
        if col is None:
            return float(fallback)
'''
new='''        col = batting_metric(df)
        if col is None:
            return float(fallback)
'''
rep(old,new,'offense metric selection',1)

old='''        team_val=float(team_rows['_v'].dropna().iloc[-1])
        league_vals=season['_v'].dropna()
        center=float(league_vals.median()) if len(league_vals) else float(x['_v'].dropna().median())
        if not center or pd.isna(center):
            return float(fallback)
        return float(np.clip((team_val/center)*100.0,75.0,125.0))
'''
new='''        team_val=float(team_rows['_v'].dropna().iloc[-1])
        # Real FanGraphs wRC+ is already league/park adjusted around 100.
        if col == 'wRC+' and 'wRC+_Source' in season.columns:
            src = season.loc[team_rows.index[-1], 'wRC+_Source'] if team_rows.index[-1] in season.index else ''
            if 'FANGRAPHS_REAL' in str(src):
                return float(np.clip(team_val,70.0,130.0))
        league_vals=season['_v'].dropna()
        center=float(league_vals.median()) if len(league_vals) else float(x['_v'].dropna().median())
        if not center or pd.isna(center):
            return float(fallback)
        return float(np.clip((team_val/center)*100.0,75.0,125.0))
'''
rep(old,new,'real wRC normalization',1)

old='''        col='ERA' if 'ERA' in match.columns else ('xFIP' if 'xFIP' in match.columns else None)
        if col is None:
            return None
        val=pd.to_numeric(match.iloc[-1][col],errors='coerce')
        return None if pd.isna(val) else float(val)
'''
new='''        value, used = row_pitching_value(match.iloc[-1], None)
        return None if value is None else float(value)
'''
rep(old,new,'starter quality metric',1)

old='''        if df.empty or 'Team' not in df.columns or 'ERA' not in df.columns:
            return pd.DataFrame()
        df['ERA']=pd.to_numeric(df['ERA'],errors='coerce')
'''
new='''        if df.empty or 'Team' not in df.columns or 'ERA' not in df.columns:
            return pd.DataFrame()
        df['ERA']=pd.to_numeric(df['ERA'],errors='coerce')
        for c in ('xFIP','FIP'):
            if c in df.columns:
                df[c]=pd.to_numeric(df[c],errors='coerce')
'''
rep(old,new,'bullpen advanced columns',1)

old="        return float(rows.iloc[-1]['ERA'])\n"
new="        value, used = row_pitching_value(rows.iloc[-1], fallback)\n        return float(value)\n"
rep(old,new,'bullpen quality metric',1)

rep("                            bat_col_ml = 'OPS_Index' if 'OPS_Index' in df_bat.columns else 'wRC+'\n                            pit_col_ml = 'ERA' if 'ERA' in df_pit.columns else 'xFIP'\n", "                            bat_col_ml = batting_metric(df_bat) or ('OPS_Index' if 'OPS_Index' in df_bat.columns else 'wRC+')\n                            pit_col_ml = pitching_metric(df_pit) or ('ERA' if 'ERA' in df_pit.columns else 'xFIP')\n", 'scanner ML metric choice',1)

# Individual mode contains the same prior-stat definition once.
rep("                    bat_col_ml = 'OPS_Index' if 'OPS_Index' in df_bat.columns else 'wRC+'\n                    pit_col_ml = 'ERA' if 'ERA' in df_pit.columns else 'xFIP'\n", "                    bat_col_ml = batting_metric(df_bat) or ('OPS_Index' if 'OPS_Index' in df_bat.columns else 'wRC+')\n                    pit_col_ml = pitching_metric(df_pit) or ('ERA' if 'ERA' in df_pit.columns else 'xFIP')\n", 'individual ML metric choice',1)

# Surface which statistical sources are actually active.
marker='''        modo_app = st.sidebar.radio("Modo de Operación", ["🎯 Análisis Individual por Partido", "🔍 Escáner Automático de la Jornada (EV+)"])
'''
replacement='''        bat_active = batting_metric(df_bat) or 'N/D'
        pit_active = pitching_metric(df_pit) or 'N/D'
        st.sidebar.caption(f"V6 métricas activas · Ofensiva: {bat_active} · Pitcheo: {pit_active}")
        modo_app = st.sidebar.radio("Modo de Operación", ["🎯 Análisis Individual por Partido", "🔍 Escáner Automático de la Jornada (EV+)"])
'''
rep(marker,replacement,'active metrics UI',1)

p.write_text(t,encoding='utf-8')
print('V6 app patch applied')
