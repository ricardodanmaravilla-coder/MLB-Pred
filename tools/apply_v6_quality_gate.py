from pathlib import Path
p=Path('app_mlb.py'); t=p.read_text(encoding='utf-8')

def rep(old,new,label,expected=1):
    global t
    n=t.count(old)
    if n!=expected: raise SystemExit(f'{label}: expected {expected}, found {n}')
    t=t.replace(old,new)

rep('''                            if xfip_loc is None:
                                team_pit_loc = df_pit[df_pit['Team'] == loc_abbr]
                                if team_pit_loc.empty:
                                    errores_datos.append({"Partido": f"{datos_partido['visita']} @ {datos_partido['local']}", "Error": "Sin pitching local para fallback"})
                                    continue
                                xfip_loc = float(team_pit_loc.iloc[-1]['xFIP'])
''','''                            if xfip_loc is None:
                                errores_datos.append({"Partido": f"{datos_partido['visita']} @ {datos_partido['local']}", "Error": f"Abridor local sin métrica individual fiable: {pitcher_loc_nombre}"})
                                continue
''','scanner local starter gate')
rep('''                            if xfip_vis is None:
                                team_pit_vis = df_pit[df_pit['Team'] == vis_abbr]
                                if team_pit_vis.empty:
                                    errores_datos.append({"Partido": f"{datos_partido['visita']} @ {datos_partido['local']}", "Error": "Sin pitching visitante para fallback"})
                                    continue
                                xfip_vis = float(team_pit_vis.iloc[-1]['xFIP'])
''','''                            if xfip_vis is None:
                                errores_datos.append({"Partido": f"{datos_partido['visita']} @ {datos_partido['local']}", "Error": f"Abridor visitante sin métrica individual fiable: {pitcher_vis_nombre}"})
                                continue
''','scanner visitor starter gate')
rep('''                    if xfip_loc is None:
                        team_pit_loc = df_pit[df_pit['Team'] == loc_abbr]
                        if team_pit_loc.empty:
                            st.error("❌ No hay datos de pitcheo reales para el local.")
                            st.stop()
                        xfip_loc = float(team_pit_loc.iloc[-1]['xFIP']) 
''','''                    if xfip_loc is None:
                        st.error(f"❌ Abridor local sin métrica individual fiable: {pitcher_loc_nombre}. No se emite apuesta con un proxy de equipo.")
                        st.stop()
''','individual local starter gate')
rep('''                    if xfip_vis is None:
                        team_pit_vis = df_pit[df_pit['Team'] == vis_abbr]
                        if team_pit_vis.empty:
                            st.error("❌ No hay datos de pitcheo reales para el visitante.")
                            st.stop()
                        xfip_vis = float(team_pit_vis.iloc[-1]['xFIP']) 
''','''                    if xfip_vis is None:
                        st.error(f"❌ Abridor visitante sin métrica individual fiable: {pitcher_vis_nombre}. No se emite apuesta con un proxy de equipo.")
                        st.stop()
''','individual visitor starter gate')
p.write_text(t,encoding='utf-8'); print('V6 starter quality gate applied')
