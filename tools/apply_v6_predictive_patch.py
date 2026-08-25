from pathlib import Path

p=Path('app_mlb.py'); t=p.read_text(encoding='utf-8')

def rep(old,new,label,expected=1):
    global t
    n=t.count(old)
    if n!=expected: raise SystemExit(f'{label}: expected {expected}, found {n}')
    t=t.replace(old,new)

rep('from modules.ml_mlb import PredictorMLMLB\n','from modules.ml_mlb import PredictorMLMLB, preferred_batting_column, preferred_pitching_column\nfrom modules.blend_calibration import market_blend_weights\n','imports')

start=t.index('def _current_offensive_index('); end=t.index('\ndef _starter_run_prevention',start)
new_off='''def _current_offensive_index(df, team, fallback=100.0):
    """Current offense centered at league average, preferring real FanGraphs wRC+."""
    try:
        if df is None or df.empty or 'Team' not in df.columns:
            return float(fallback)
        col = preferred_batting_column(df)
        if col is None:
            return float(fallback)
        x=df.copy(); x['_v']=pd.to_numeric(x[col],errors='coerce')
        if 'Season' in x.columns:
            x['_s']=pd.to_numeric(x['Season'],errors='coerce'); latest=x['_s'].dropna().max(); season=x[x['_s']==latest]
        else:
            season=x
        team_rows=season[season['Team']==team]
        if team_rows.empty: team_rows=x[x['Team']==team]
        vals=team_rows['_v'].dropna()
        if vals.empty: return float(fallback)
        team_val=float(vals.iloc[-1])
        if col == 'wRC+':
            return float(np.clip(team_val, 70.0, 130.0))
        league=season['_v'].dropna(); center=float(league.median()) if len(league) else float(x['_v'].dropna().median())
        if not center or pd.isna(center): return float(fallback)
        return float(np.clip((team_val/center)*100.0,75.0,125.0))
    except Exception:
        return float(fallback)

'''
t=t[:start]+new_off+t[end+1:]

rep("        col='ERA' if 'ERA' in match.columns else ('xFIP' if 'xFIP' in match.columns else None)\n","        col=next((c for c in ('xFIP','FIP','ERA') if c in match.columns and pd.to_numeric(match[c],errors='coerce').notna().any()), None)\n",'starter metric')
rep("st.title(\"⚾ MLB Quant Analytics Pro (Montecarlo + Kelly)\")","st.title(\"⚾ MLB Quant Analytics Pro V6\")",'title')
rep('                    recomendaciones = []\n                    diagnostico_totales = []\n','                    recomendaciones = []\n                    diagnostico_totales = []\n                    blend_weights = market_blend_weights()\n','scanner weights')
rep("                            bat_col_ml = 'OPS_Index' if 'OPS_Index' in df_bat.columns else 'wRC+'\n                            pit_col_ml = 'ERA' if 'ERA' in df_pit.columns else 'xFIP'\n","                            bat_col_ml = preferred_batting_column(df_bat) or 'OPS_Index'\n                            pit_col_ml = preferred_pitching_column(df_pit) or 'ERA'\n",'scanner metric preference')
rep("                    bat_col_ml = 'OPS_Index' if 'OPS_Index' in df_bat.columns else 'wRC+'\n                    pit_col_ml = 'ERA' if 'ERA' in df_pit.columns else 'xFIP'\n","                    bat_col_ml = preferred_batting_column(df_bat) or 'OPS_Index'\n                    pit_col_ml = preferred_pitching_column(df_pit) or 'ERA'\n",'individual metric preference')

# Scanner candidates.
repls={
"moneyline_candidate(f\"Gana Local ({datos_partido['local']})\", prob_ml_loc, prob_mc_loc, cuota_loc, mkt_loc_scanner)":"moneyline_candidate(f\"Gana Local ({datos_partido['local']})\", prob_ml_loc, prob_mc_loc, cuota_loc, mkt_loc_scanner, blend_weight_ml=blend_weights['Moneyline']['ml_weight'])",
"moneyline_candidate(f\"Gana Visita ({datos_partido['visita']})\", prob_ml_vis, prob_mc_vis, cuota_vis, mkt_vis_scanner)":"moneyline_candidate(f\"Gana Visita ({datos_partido['visita']})\", prob_ml_vis, prob_mc_vis, cuota_vis, mkt_vis_scanner, blend_weight_ml=blend_weights['Moneyline']['ml_weight'])",
"total_candidate(f\"Over {linea_casino}\", prob_ml_over, prob_mc_over, datos_partido.get(\"cuota_over\"), mkt_over_scanner, carreras_dict.get(f\"Push {linea_casino}\", 0.0))":"total_candidate(f\"Over {linea_casino}\", prob_ml_over, prob_mc_over, datos_partido.get(\"cuota_over\"), mkt_over_scanner, carreras_dict.get(f\"Push {linea_casino}\", 0.0), blend_weight_ml=blend_weights['Totales']['ml_weight'])",
"total_candidate(f\"Under {linea_casino}\", prob_ml_under, prob_mc_under, datos_partido.get(\"cuota_under\"), mkt_under_scanner, carreras_dict.get(f\"Push {linea_casino}\", 0.0))":"total_candidate(f\"Under {linea_casino}\", prob_ml_under, prob_mc_under, datos_partido.get(\"cuota_under\"), mkt_under_scanner, carreras_dict.get(f\"Push {linea_casino}\", 0.0), blend_weight_ml=blend_weights['Totales']['ml_weight'])",
"runline_candidate(f\"Hándicap {spread_loc:+.1f} ({datos_partido['local']})\", prob_ml_spread_loc, prob_mc_spread_loc, datos_partido.get(\"cuota_spread_loc\"), mkt_sp_loc_scanner, carreras_dict.get(f\"Push Spread Local {spread_loc:+.1f}\", 0.0))":"runline_candidate(f\"Hándicap {spread_loc:+.1f} ({datos_partido['local']})\", prob_ml_spread_loc, prob_mc_spread_loc, datos_partido.get(\"cuota_spread_loc\"), mkt_sp_loc_scanner, carreras_dict.get(f\"Push Spread Local {spread_loc:+.1f}\", 0.0), blend_weight_ml=blend_weights['Hándicap']['ml_weight'])",
"runline_candidate(f\"Hándicap {spread_vis:+.1f} ({datos_partido['visita']})\", prob_ml_spread_vis, prob_mc_spread_vis, datos_partido.get(\"cuota_spread_vis\"), mkt_sp_vis_scanner, carreras_dict.get(f\"Push Spread Visita {spread_vis:+.1f}\", 0.0))":"runline_candidate(f\"Hándicap {spread_vis:+.1f} ({datos_partido['visita']})\", prob_ml_spread_vis, prob_mc_spread_vis, datos_partido.get(\"cuota_spread_vis\"), mkt_sp_vis_scanner, carreras_dict.get(f\"Push Spread Visita {spread_vis:+.1f}\", 0.0), blend_weight_ml=blend_weights['Hándicap']['ml_weight'])",
}
for old,new in repls.items(): rep(old,new,'scanner candidate')

# Individual weights and candidates.
rep('                    candidatos_ind = [\n','                    blend_weights = market_blend_weights()\n                    candidatos_ind = [\n','individual weights')
repls2={
"moneyline_candidate(f\"Gana Local ({datos_partido['local']})\", prob_ml_loc, prob_mc_loc, cuotas_reales['Moneyline_Local'], mkt_loc)":"moneyline_candidate(f\"Gana Local ({datos_partido['local']})\", prob_ml_loc, prob_mc_loc, cuotas_reales['Moneyline_Local'], mkt_loc, blend_weight_ml=blend_weights['Moneyline']['ml_weight'])",
"moneyline_candidate(f\"Gana Visita ({datos_partido['visita']})\", prob_ml_vis, prob_mc_vis, cuotas_reales['Moneyline_Visita'], mkt_vis)":"moneyline_candidate(f\"Gana Visita ({datos_partido['visita']})\", prob_ml_vis, prob_mc_vis, cuotas_reales['Moneyline_Visita'], mkt_vis, blend_weight_ml=blend_weights['Moneyline']['ml_weight'])",
"total_candidate(f\"Over {linea_casino}\", prob_ml_over, prob_mc_over, cuotas_reales['Cuota_Over'], mkt_over, carreras_dict.get(f\"Push {linea_casino}\", 0.0))":"total_candidate(f\"Over {linea_casino}\", prob_ml_over, prob_mc_over, cuotas_reales['Cuota_Over'], mkt_over, carreras_dict.get(f\"Push {linea_casino}\", 0.0), blend_weight_ml=blend_weights['Totales']['ml_weight'])",
"total_candidate(f\"Under {linea_casino}\", prob_ml_under, prob_mc_under, cuotas_reales['Cuota_Under'], mkt_under, carreras_dict.get(f\"Push {linea_casino}\", 0.0))":"total_candidate(f\"Under {linea_casino}\", prob_ml_under, prob_mc_under, cuotas_reales['Cuota_Under'], mkt_under, carreras_dict.get(f\"Push {linea_casino}\", 0.0), blend_weight_ml=blend_weights['Totales']['ml_weight'])",
"runline_candidate(f\"Hándicap {spread_loc:+.1f} ({datos_partido['local']})\", prob_ml_spread_loc, prob_mc_spread_loc, cuotas_reales['Cuota_Spread_Local'], mkt_sp_loc, carreras_dict.get(f\"Push Spread Local {spread_loc:+.1f}\", 0.0))":"runline_candidate(f\"Hándicap {spread_loc:+.1f} ({datos_partido['local']})\", prob_ml_spread_loc, prob_mc_spread_loc, cuotas_reales['Cuota_Spread_Local'], mkt_sp_loc, carreras_dict.get(f\"Push Spread Local {spread_loc:+.1f}\", 0.0), blend_weight_ml=blend_weights['Hándicap']['ml_weight'])",
"runline_candidate(f\"Hándicap {spread_vis:+.1f} ({datos_partido['visita']})\", prob_ml_spread_vis, prob_mc_spread_vis, cuotas_reales['Cuota_Spread_Visita'], mkt_sp_vis, carreras_dict.get(f\"Push Spread Visita {spread_vis:+.1f}\", 0.0))":"runline_candidate(f\"Hándicap {spread_vis:+.1f} ({datos_partido['visita']})\", prob_ml_spread_vis, prob_mc_spread_vis, cuotas_reales['Cuota_Spread_Visita'], mkt_sp_vis, carreras_dict.get(f\"Push Spread Visita {spread_vis:+.1f}\", 0.0), blend_weight_ml=blend_weights['Hándicap']['ml_weight'])",
}
for old,new in repls2.items(): rep(old,new,'individual candidate')
rep("                            'Prob. Combinada': f\"{round(cand.probability, 1)}%\",\n","                            'Prob. Combinada': f\"{round(cand.probability, 1)}%\",\n                            'Peso ML': f\"{round(cand.blend_weight_ml*100, 0)}%\",\n",'blend display')
t=t.replace("'model_version': 'v5'","'model_version': 'v6'")
p.write_text(t,encoding='utf-8'); print('V6 predictive app patch applied')
