from pathlib import Path

p=Path('app_mlb.py')
t=p.read_text(encoding='utf-8')

def rep(old,new,label,expected):
    global t
    n=t.count(old)
    if n!=expected:
        raise SystemExit(f'{label}: expected {expected}, found {n}')
    t=t.replace(old,new)

rep("""                                xfip_loc = float(team_pit_loc.iloc[-1]['xFIP'])
""","""                                xfip_loc, _ = row_pitching_value(team_pit_loc.iloc[-1], None)
                                if xfip_loc is None:
                                    errores_datos.append({"Partido": f"{datos_partido['visita']} @ {datos_partido['local']}", "Error": "Sin métrica válida de pitching local"})
                                    continue
""",'scanner home fallback',1)
rep("""                                xfip_vis = float(team_pit_vis.iloc[-1]['xFIP'])
""","""                                xfip_vis, _ = row_pitching_value(team_pit_vis.iloc[-1], None)
                                if xfip_vis is None:
                                    errores_datos.append({"Partido": f"{datos_partido['visita']} @ {datos_partido['local']}", "Error": "Sin métrica válida de pitching visitante"})
                                    continue
""",'scanner away fallback',1)

rep("""                        xfip_loc = float(team_pit_loc.iloc[-1]['xFIP']) 
""","""                        xfip_loc, _ = row_pitching_value(team_pit_loc.iloc[-1], None)
                        if xfip_loc is None:
                            st.error("❌ No hay una métrica válida de pitcheo para el local.")
                            st.stop()
""",'individual home fallback',1)
rep("""                        xfip_vis = float(team_pit_vis.iloc[-1]['xFIP']) 
""","""                        xfip_vis, _ = row_pitching_value(team_pit_vis.iloc[-1], None)
                        if xfip_vis is None:
                            st.error("❌ No hay una métrica válida de pitcheo para el visitante.")
                            st.stop()
""",'individual away fallback',1)
rep("'model_version': 'v5'", "'model_version': 'v6'", 'ledger version',1)

p.write_text(t,encoding='utf-8')
print('V6 live fallback patch applied')
