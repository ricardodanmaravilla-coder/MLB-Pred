from pathlib import Path

p=Path('app_mlb.py')
s=p.read_text(encoding='utf-8')
orig=s

marker='''def calcular_criterio_kelly(probabilidad_real, cuota_decimal, fraccion=0.25):\n'''
helper='''def _current_offensive_index(df, team, fallback=100.0):\n    """Return current team offense centered at league average = 100.\n\n    The repository legacy wRC+ column is OPS*100, not real wRC+. Monte Carlo\n    needs a centered multiplier, so normalize the latest-season OPS index by\n    that season's league median before passing it to the simulator.\n    """\n    try:\n        if df is None or df.empty or 'Team' not in df.columns:\n            return float(fallback)\n        col = 'OPS_Index' if 'OPS_Index' in df.columns else ('wRC+' if 'wRC+' in df.columns else None)\n        if col is None:\n            return float(fallback)\n        x=df.copy(); x['_v']=pd.to_numeric(x[col],errors='coerce')\n        if 'Season' in x.columns:\n            x['_s']=pd.to_numeric(x['Season'],errors='coerce')\n            latest=x['_s'].dropna().max()\n            season=x[x['_s']==latest]\n        else:\n            season=x\n        team_rows=season[season['Team']==team]\n        if team_rows.empty:\n            team_rows=x[x['Team']==team]\n        if team_rows.empty:\n            return float(fallback)\n        team_val=float(team_rows['_v'].dropna().iloc[-1])\n        league_vals=season['_v'].dropna()\n        center=float(league_vals.median()) if len(league_vals) else float(x['_v'].dropna().median())\n        if not center or pd.isna(center):\n            return float(fallback)\n        return float(np.clip((team_val/center)*100.0,75.0,125.0))\n    except Exception:\n        return float(fallback)\n\n\ndef _starter_run_prevention(df, pitcher_name):\n    """Resolve a starter safely. The legacy xFIP column currently contains ERA.\n\n    Prefer exact full-name matching. A last-name fallback is allowed only when\n    it identifies one unique player, preventing accidental matches for common surnames.\n    """\n    try:\n        if not pitcher_name or pitcher_name == 'Por Anunciar' or df is None or df.empty or 'Name' not in df.columns:\n            return None\n        names=df['Name'].astype(str)\n        exact=df[names.str.casefold()==str(pitcher_name).casefold()]\n        match=exact\n        if match.empty:\n            last=str(pitcher_name).split()[-1].casefold()\n            fallback=df[names.str.split().str[-1].str.casefold()==last]\n            if fallback['Name'].nunique()!=1:\n                return None\n            match=fallback\n        col='ERA' if 'ERA' in match.columns else ('xFIP' if 'xFIP' in match.columns else None)\n        if col is None:\n            return None\n        val=pd.to_numeric(match.iloc[-1][col],errors='coerce')\n        return None if pd.isna(val) else float(val)\n    except Exception:\n        return None\n\n\n'''
if 'def _current_offensive_index(' not in s:
    s=s.replace(marker,helper+marker,1)

# Scanner offense: normalize legacy OPS index to league-average 100.
old_sc='''                            team_bat_loc = df_bat[df_bat['Team'] == loc_abbr]\n                            wrc_loc = float(team_bat_loc.iloc[-1]['wRC+']) if not team_bat_loc.empty else 100.0\n                            \n                            team_bat_vis = df_bat[df_bat['Team'] == vis_abbr]\n                            wrc_vis = float(team_bat_vis.iloc[-1]['wRC+']) if not team_bat_vis.empty else 100.0\n                            \n                            pitcher_loc_nombre = datos_partido["pitcher_local"]\n                            xfip_loc = None\n                            if pitcher_loc_nombre != "Por Anunciar" and not df_pit_ind.empty:\n                                match_loc = df_pit_ind[df_pit_ind['Name'].str.contains(pitcher_loc_nombre.split()[-1], case=False, na=False)]\n                                if not match_loc.empty:\n                                    xfip_loc = float(match_loc.iloc[-1]['xFIP'])\n'''
new_sc='''                            wrc_loc = _current_offensive_index(df_bat, loc_abbr)\n                            wrc_vis = _current_offensive_index(df_bat, vis_abbr)\n                            \n                            pitcher_loc_nombre = datos_partido["pitcher_local"]\n                            xfip_loc = _starter_run_prevention(df_pit_ind, pitcher_loc_nombre)\n'''
if old_sc not in s:
    raise SystemExit('scanner offense/starter marker not found')
s=s.replace(old_sc,new_sc,1)

old_sc_vis='''                            pitcher_vis_nombre = datos_partido["pitcher_visita"]\n                            xfip_vis = None\n                            if pitcher_vis_nombre != "Por Anunciar" and not df_pit_ind.empty:\n                                match_vis = df_pit_ind[df_pit_ind['Name'].str.contains(pitcher_vis_nombre.split()[-1], case=False, na=False)]\n                                if not match_vis.empty:\n                                    xfip_vis = float(match_vis.iloc[-1]['xFIP'])\n'''
new_sc_vis='''                            pitcher_vis_nombre = datos_partido["pitcher_visita"]\n                            xfip_vis = _starter_run_prevention(df_pit_ind, pitcher_vis_nombre)\n'''
if old_sc_vis not in s:
    raise SystemExit('scanner visitor starter marker not found')
s=s.replace(old_sc_vis,new_sc_vis,1)

# Individual offense block.
old_ind='''                    try:\n                        team_bat_loc = df_bat[df_bat['Team'] == loc_abbr]\n                        wrc_loc = float(team_bat_loc.iloc[-1]['wRC+']) if not team_bat_loc.empty else 100.0\n                        \n                        team_bat_vis = df_bat[df_bat['Team'] == vis_abbr]\n                        wrc_vis = float(team_bat_vis.iloc[-1]['wRC+']) if not team_bat_vis.empty else 100.0\n                    except Exception as e:\n                        st.error(f"Error procesando wRC+ de bateo: {e}")\n                        st.stop()\n                    \n                    pitcher_loc_nombre = datos_partido["pitcher_local"]\n                    xfip_loc = None\n                    if pitcher_loc_nombre != "Por Anunciar" and not df_pit_ind.empty:\n                        match_loc = df_pit_ind[df_pit_ind['Name'].str.contains(pitcher_loc_nombre.split()[-1], case=False, na=False)]\n                        if not match_loc.empty:\n                            xfip_loc = float(match_loc.iloc[-1]['xFIP']) \n'''
new_ind='''                    try:\n                        wrc_loc = _current_offensive_index(df_bat, loc_abbr)\n                        wrc_vis = _current_offensive_index(df_bat, vis_abbr)\n                    except Exception as e:\n                        st.error(f"Error procesando índice ofensivo: {e}")\n                        st.stop()\n                    \n                    pitcher_loc_nombre = datos_partido["pitcher_local"]\n                    xfip_loc = _starter_run_prevention(df_pit_ind, pitcher_loc_nombre)\n'''
if old_ind not in s:
    raise SystemExit('individual offense/starter marker not found')
s=s.replace(old_ind,new_ind,1)

old_ind_vis='''                    pitcher_vis_nombre = datos_partido["pitcher_visita"]\n                    xfip_vis = None\n                    if pitcher_vis_nombre != "Por Anunciar" and not df_pit_ind.empty:\n                        match_vis = df_pit_ind[df_pit_ind['Name'].str.contains(pitcher_vis_nombre.split()[-1], case=False, na=False)]\n                        if not match_vis.empty:\n                            xfip_vis = float(match_vis.iloc[-1]['xFIP']) \n'''
new_ind_vis='''                    pitcher_vis_nombre = datos_partido["pitcher_visita"]\n                    xfip_vis = _starter_run_prevention(df_pit_ind, pitcher_vis_nombre)\n'''
if old_ind_vis not in s:
    raise SystemExit('individual visitor starter marker not found')
s=s.replace(old_ind_vis,new_ind_vis,1)

if s==orig:
    raise SystemExit('no changes')
for req in ['def _current_offensive_index','def _starter_run_prevention','_current_offensive_index(df_bat, loc_abbr)','_starter_run_prevention(df_pit_ind, pitcher_loc_nombre)']:
    if req not in s:
        raise SystemExit(f'missing {req}')
p.write_text(s,encoding='utf-8')
print('offense/starter V3 patch applied')
