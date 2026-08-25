from pathlib import Path

# Monte Carlo metadata remains True because even a real season reliever aggregate is
# still a proxy for today's available bullpen.
pm=Path('modules/montecarlo_mlb.py'); mt=pm.read_text(encoding='utf-8')
old="'Pitching_Agregado_Es_Proxy_Bullpen':False"
if mt.count(old)!=1: raise SystemExit(f'MC metadata marker found {mt.count(old)} times')
mt=mt.replace(old,"'Pitching_Agregado_Es_Proxy_Bullpen':True")
pm.write_text(mt,encoding='utf-8')

p=Path('app_mlb.py'); t=p.read_text(encoding='utf-8')
marker='''def _starter_run_prevention(df, pitcher_name):'''
start=t.index(marker); next_pos=t.index('\n\n@st.cache_data',start)
block=t[start:next_pos]
addition='''\n\ndef _starter_expected_innings(df, pitcher_name, default=5.2):
    """Expected starter workload from season IP/GS, safely bounded."""
    try:
        if not pitcher_name or pitcher_name == 'Por Anunciar' or df is None or df.empty or 'Name' not in df.columns:
            return float(default)
        names=df['Name'].astype(str); match=df[names.str.casefold()==str(pitcher_name).casefold()]
        if match.empty:
            last=str(pitcher_name).split()[-1].casefold(); alt=df[names.str.split().str[-1].str.casefold()==last]
            if alt['Name'].nunique()!=1: return float(default)
            match=alt
        ip=pd.to_numeric(match.get('IP'),errors='coerce') if 'IP' in match.columns else pd.Series(dtype=float)
        gs=pd.to_numeric(match.get('GS'),errors='coerce') if 'GS' in match.columns else pd.Series(dtype=float)
        if len(ip) and len(gs) and pd.notna(ip.iloc[-1]) and pd.notna(gs.iloc[-1]) and float(gs.iloc[-1])>0:
            return float(np.clip(float(ip.iloc[-1])/float(gs.iloc[-1]),3.5,6.8))
        return float(default)
    except Exception:
        return float(default)
'''
if '_starter_expected_innings' not in t: t=t[:next_pos]+addition+t[next_pos:]

# Add workload resolution after starter run-prevention lines in scanner and individual.
old1='''                            xfip_loc = _starter_run_prevention(df_pit_ind, pitcher_loc_nombre)
                            
                            if xfip_loc is None:'''
new1='''                            xfip_loc = _starter_run_prevention(df_pit_ind, pitcher_loc_nombre)
                            starter_ip_loc = _starter_expected_innings(df_pit_ind, pitcher_loc_nombre)
                            
                            if xfip_loc is None:'''
if t.count(old1)!=1: raise SystemExit(f'scanner loc starter marker {t.count(old1)}')
t=t.replace(old1,new1)
old2='''                            xfip_vis = _starter_run_prevention(df_pit_ind, pitcher_vis_nombre)
                            
                            if xfip_vis is None:'''
new2='''                            xfip_vis = _starter_run_prevention(df_pit_ind, pitcher_vis_nombre)
                            starter_ip_vis = _starter_expected_innings(df_pit_ind, pitcher_vis_nombre)
                            
                            if xfip_vis is None:'''
if t.count(old2)!=1: raise SystemExit(f'scanner vis starter marker {t.count(old2)}')
t=t.replace(old2,new2)
old3='''                    xfip_loc = _starter_run_prevention(df_pit_ind, pitcher_loc_nombre)
                    
                    if xfip_loc is None:'''
new3='''                    xfip_loc = _starter_run_prevention(df_pit_ind, pitcher_loc_nombre)
                    starter_ip_loc = _starter_expected_innings(df_pit_ind, pitcher_loc_nombre)
                    
                    if xfip_loc is None:'''
if t.count(old3)!=1: raise SystemExit(f'ind loc starter marker {t.count(old3)}')
t=t.replace(old3,new3)
old4='''                    xfip_vis = _starter_run_prevention(df_pit_ind, pitcher_vis_nombre)
                    
                    if xfip_vis is None:'''
new4='''                    xfip_vis = _starter_run_prevention(df_pit_ind, pitcher_vis_nombre)
                    starter_ip_vis = _starter_expected_innings(df_pit_ind, pitcher_vis_nombre)
                    
                    if xfip_vis is None:'''
if t.count(old4)!=1: raise SystemExit(f'ind vis starter marker {t.count(old4)}')
t=t.replace(old4,new4)
# Add optional args to both MC calls.
needle='''                                df_games=df_games,
                                num_simulaciones=50000
                            )'''
repl='''                                df_games=df_games,
                                num_simulaciones=50000,
                                starter_ip_loc=starter_ip_loc, starter_ip_vis=starter_ip_vis
                            )'''
if t.count(needle)!=1: raise SystemExit(f'scanner MC marker {t.count(needle)}')
t=t.replace(needle,repl)
needle2='''                        df_games=df_games,
                        num_simulaciones=50000
                    )'''
repl2='''                        df_games=df_games,
                        num_simulaciones=50000,
                        starter_ip_loc=starter_ip_loc, starter_ip_vis=starter_ip_vis
                    )'''
if t.count(needle2)!=1: raise SystemExit(f'ind MC marker {t.count(needle2)}')
t=t.replace(needle2,repl2)
p.write_text(t,encoding='utf-8'); print('V6 starter workload patch applied')
