from pathlib import Path

p = Path('app_mlb.py')
t = p.read_text(encoding='utf-8')


def rep(old, new, label, expected=1):
    global t
    n = t.count(old)
    if n != expected:
        raise SystemExit(f'{label}: expected {expected}, found {n}')
    t = t.replace(old, new)

# Prevent duplicate labels overwriting doubleheaders in the in-memory slate.
rep(
'''                        llave = f"⚾ {away} ({away_pitcher}) @ {home} ({home_pitcher})"
''',
'''                        game_pk = game.get("gamePk")
                        llave = f"⚾ {away} ({away_pitcher}) @ {home} ({home_pitcher}) · #{game_pk}"
''',
'doubleheader key')

# Log instead of silently dropping malformed slate rows.
rep(
'''                        if not loc_abbr or not vis_abbr: continue
''',
'''                        if not loc_abbr or not vis_abbr:
                            errores_datos.append({"Partido": f"{datos_partido.get('visita','?')} @ {datos_partido.get('local','?')}", "Error": "Equipo no normalizable"})
                            continue
''',
'unknown team diagnostics')

rep(
'''                        if cuota_loc is None or cuota_vis is None or linea_casino is None:
                            continue
''',
'''                        if cuota_loc is None or cuota_vis is None or linea_casino is None:
                            errores_datos.append({"Partido": f"{datos_partido['visita']} @ {datos_partido['local']}", "Error": "Cuotas/total no disponibles o no emparejados de forma segura"})
                            continue
''',
'missing odds diagnostics')

rep(
'''                                if team_pit_loc.empty: continue
''',
'''                                if team_pit_loc.empty:
                                    errores_datos.append({"Partido": f"{datos_partido['visita']} @ {datos_partido['local']}", "Error": "Sin pitching local para fallback"})
                                    continue
''',
'home pitching diagnostics')
rep(
'''                                if team_pit_vis.empty: continue
''',
'''                                if team_pit_vis.empty:
                                    errores_datos.append({"Partido": f"{datos_partido['visita']} @ {datos_partido['local']}", "Error": "Sin pitching visitante para fallback"})
                                    continue
''',
'away pitching diagnostics')

# Carry game identity and weather provenance into the hidden scanner state.
rep(
'''                                    "_Home": datos_partido['local'], "_Away": datos_partido['visita'],
                                    "_Line": market_line, "_ProbCombined": cand.probability,
''',
'''                                    "_Home": datos_partido['local'], "_Away": datos_partido['visita'],
                                    "_GamePk": datos_partido.get('game_pk'),
                                    "_Line": market_line, "_ProbCombined": cand.probability,
''',
'game pk recommendation state')
rep(
'''                                    "_Park": park_factor, "_Temp": temp_scan, "_Wind": viento_scan, "_WindDir": dir_scan,
''',
'''                                    "_Park": park_factor, "_Temp": temp_scan, "_Wind": viento_scan, "_WindDir": dir_scan,
                                    "_WeatherSource": weather_source,
''',
'weather source state')

rep(
'''                                'game_date': slate_date().isoformat(), 'away': rr['_Away'], 'home': rr['_Home'],
''',
'''                                'game_date': slate_date().isoformat(), 'game_pk': rr['_GamePk'], 'away': rr['_Away'], 'home': rr['_Home'],
''',
'ledger game pk')

rep(
'''            st.caption("ℹ️ Ledger en modo local: configura GITHUB_TOKEN de escritura para persistencia entre reinicios.")
''',
'''            st.caption("ℹ️ Ledger en modo local: configura GITHUB_TOKEN y LEDGER_GITHUB_REPO para persistencia entre reinicios.")
''',
'ledger config message')

p.write_text(t, encoding='utf-8')
print('V5 follow-up app patch applied')
