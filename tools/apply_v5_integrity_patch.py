from pathlib import Path
import re


def must_replace(text, old, new, label, count=None):
    n = text.count(old)
    if n == 0:
        raise SystemExit(f'{label}: marker not found')
    if count is not None and n != count:
        raise SystemExit(f'{label}: expected {count}, found {n}')
    return text.replace(old, new)


# ---------------- app_mlb.py ----------------
p = Path('app_mlb.py')
t = p.read_text(encoding='utf-8')

t = must_replace(t,
"from modules.pick_ledger import append_snapshot\n",
"from modules.pick_ledger import append_snapshot, persistent_backend_available\nfrom modules.game_context import (\n    slate_date, park_for_team, match_odds_game, market_from_event, conservative_auto_weather\n)\n",
'import game context', 1)

t = must_replace(t,
'    return "f9ffe1d7530a88b08e853659466c46ff"\n',
'    return ""\n',
'remove public odds key', 1)

t = must_replace(t, 'target = datetime.date.today().year - 1', 'target = slate_date().year - 1', 'prior season timezone', 1)
t = must_replace(t, "hoy = datetime.date.today().strftime('%Y-%m-%d')", "hoy = slate_date().strftime('%Y-%m-%d')", 'slate timezone', 1)
t = must_replace(t, "'game_date': datetime.date.today().isoformat()", "'game_date': slate_date().isoformat()", 'ledger slate date', 1)
t = t.replace("'model_version': 'v3'", "'model_version': 'v5'")

t = must_replace(t,
'                            "pitcher_local": home_pitcher, "pitcher_visita": away_pitcher,\n',
'                            "pitcher_local": home_pitcher, "pitcher_visita": away_pitcher,\n                            "game_pk": game.get("gamePk"), "start_time_utc": game.get("gameDate"),\n',
'game identity', 1)

pattern = re.compile(r'                for item in data_odds:\n.*?            elif res_odds\.status_code == 429:', re.S)
m = pattern.search(t)
if not m:
    raise SystemExit('odds matching block not found')
new_odds = '''                for p_game in partidos.values():
                    matched_event = match_odds_game(data_odds, p_game)
                    if matched_event is None:
                        continue
                    p_game.update(market_from_event(matched_event, american_to_decimal))
            elif res_odds.status_code == 429:'''
t = t[:m.start()] + new_odds + t[m.end():]

t = must_replace(t,
'            st.markdown("Escanea la cartelera exigiendo que **Tanto Machine Learning COMO Montecarlo** tengan >60% de probabilidad de forma individual.")',
'            st.markdown("Escanea la cartelera con filtros calibrados por mercado: consenso ML + Monte Carlo, no-vig, edge, EV y desacuerdo máximo.")',
'UI filter text', 1)

t = must_replace(t,
'                    diagnostico_totales = []\n',
'                    diagnostico_totales = []\n                    errores_datos = []\n',
'data diagnostics list', 1)

scanner_park_old = '''                            df_parks.columns = df_parks.columns.str.strip()
                            park_data = pd.DataFrame()

                            col_equipo_park = next((c for c in ['Team', 'TeamCode', 'Abbr', 'Franchise', 'Equipo', 'franchise'] if c in df_parks.columns), df_parks.columns[0])
                            col_pf = next((c for c in df_parks.columns if 'park_factor' in c.lower() or 'factor' in c.lower() or 'pf' in c.lower()), None)
                            col_alt = next((c for c in df_parks.columns if 'altitud' in c.lower() or 'alt' in c.lower() or 'elevation' in c.lower() or 'pie' in c.lower()), None)

                            if not col_pf or not col_alt: continue

                            park_data = df_parks[df_parks[col_equipo_park].astype(str).str.upper() == loc_abbr.upper()]

                            if park_data.empty:
                                nombre_ciudad = datos_partido["local"].split()[-1]
                                park_data = df_parks[df_parks.apply(lambda row: row.astype(str).str.contains(nombre_ciudad, case=False).any(), axis=1)]

                            if park_data.empty: continue

                            park_factor = float(park_data[col_pf].values[0])
                            altitud = float(park_data[col_alt].values[0])
'''
scanner_park_new = '''                            park_info = park_for_team(df_parks, loc_abbr)
                            if not park_info:
                                errores_datos.append({"Partido": f"{datos_partido['visita']} @ {datos_partido['local']}", "Error": "Parque no resoluble"})
                                continue
                            park_factor = park_info['park_factor']
                            altitud = park_info['altitude_ft']
'''
t = must_replace(t, scanner_park_old, scanner_park_new, 'scanner park resolution', 1)

individual_park_old = '''                    df_parks.columns = df_parks.columns.str.strip()
                    park_data = pd.DataFrame()

                    col_equipo_park = next((c for c in ['Team', 'TeamCode', 'Abbr', 'Franchise', 'Equipo', 'franchise'] if c in df_parks.columns), df_parks.columns[0])
                    col_pf = next((c for c in df_parks.columns if 'park_factor' in c.lower() or 'factor' in c.lower() or 'pf' in c.lower()), None)
                    col_alt = next((c for c in df_parks.columns if 'altitud' in c.lower() or 'alt' in c.lower() or 'elevation' in c.lower() or 'pie' in c.lower()), None)

                    if not col_pf or not col_alt:
                        st.error("❌ El archivo `mlb_park_factors.csv` no contiene columnas reconocibles de Park Factor o Altitud.")
                        st.stop()

                    park_data = df_parks[df_parks[col_equipo_park].astype(str).str.upper() == loc_abbr.upper()]

                    if park_data.empty:
                        nombre_ciudad = datos_partido["local"].split()[-1]
                        park_data = df_parks[df_parks.apply(lambda row: row.astype(str).str.contains(nombre_ciudad, case=False).any(), axis=1)]

                    if park_data.empty:
                        st.error(f"❌ Error de integridad: No se encontró ningún registro real para el equipo '{datos_partido['local']}' ({loc_abbr}) en 'mlb_park_factors.csv'. Verifica tu archivo de estadios.")
                        st.stop()

                    park_factor = float(park_data[col_pf].values[0])
                    altitud = float(park_data[col_alt].values[0])
'''
individual_park_new = '''                    park_info = park_for_team(df_parks, loc_abbr)
                    if not park_info:
                        st.error(f"❌ Error de integridad: no se pudo resolver el parque de {datos_partido['local']} ({loc_abbr}).")
                        st.stop()
                    park_factor = park_info['park_factor']
                    altitud = park_info['altitude_ft']
'''
t = must_replace(t, individual_park_old, individual_park_new, 'individual park resolution', 1)

weather_old = '''                            temp_scan, viento_scan, dir_scan = obtener_clima_estadio(datos_partido["local"])
                            temp_scan = 72 if temp_scan is None else temp_scan
                            viento_scan = 8 if viento_scan is None else viento_scan
                            dir_scan = "None" if not dir_scan else dir_scan
'''
weather_new = '''                            temp_raw, viento_raw, dir_raw = obtener_clima_estadio(datos_partido["local"])
                            temp_scan, viento_scan, dir_scan, weather_source = conservative_auto_weather(
                                datos_partido["local"], datos_partido.get("start_time_utc"), temp_raw, viento_raw, dir_raw
                            )
'''
t = must_replace(t, weather_old, weather_new, 'weather guardrail', 1)

# Push-aware candidate construction, scanner and individual.
repls = {
'''total_candidate(f"Over {linea_casino}", prob_ml_over, prob_mc_over, datos_partido.get("cuota_over"), mkt_over_scanner)''':
'''total_candidate(f"Over {linea_casino}", prob_ml_over, prob_mc_over, datos_partido.get("cuota_over"), mkt_over_scanner, carreras_dict.get(f"Push {linea_casino}", 0.0))''',
'''total_candidate(f"Under {linea_casino}", prob_ml_under, prob_mc_under, datos_partido.get("cuota_under"), mkt_under_scanner)''':
'''total_candidate(f"Under {linea_casino}", prob_ml_under, prob_mc_under, datos_partido.get("cuota_under"), mkt_under_scanner, carreras_dict.get(f"Push {linea_casino}", 0.0))''',
'''runline_candidate(f"Hándicap {spread_loc:+.1f} ({datos_partido['local']})", prob_ml_spread_loc, prob_mc_spread_loc, datos_partido.get("cuota_spread_loc"), mkt_sp_loc_scanner)''':
'''runline_candidate(f"Hándicap {spread_loc:+.1f} ({datos_partido['local']})", prob_ml_spread_loc, prob_mc_spread_loc, datos_partido.get("cuota_spread_loc"), mkt_sp_loc_scanner, carreras_dict.get(f"Push Spread Local {spread_loc:+.1f}", 0.0))''',
'''runline_candidate(f"Hándicap {spread_vis:+.1f} ({datos_partido['visita']})", prob_ml_spread_vis, prob_mc_spread_vis, datos_partido.get("cuota_spread_vis"), mkt_sp_vis_scanner)''':
'''runline_candidate(f"Hándicap {spread_vis:+.1f} ({datos_partido['visita']})", prob_ml_spread_vis, prob_mc_spread_vis, datos_partido.get("cuota_spread_vis"), mkt_sp_vis_scanner, carreras_dict.get(f"Push Spread Visita {spread_vis:+.1f}", 0.0))''',
'''total_candidate(f"Over {linea_casino}", prob_ml_over, prob_mc_over, cuotas_reales['Cuota_Over'], mkt_over)''':
'''total_candidate(f"Over {linea_casino}", prob_ml_over, prob_mc_over, cuotas_reales['Cuota_Over'], mkt_over, carreras_dict.get(f"Push {linea_casino}", 0.0))''',
'''total_candidate(f"Under {linea_casino}", prob_ml_under, prob_mc_under, cuotas_reales['Cuota_Under'], mkt_under)''':
'''total_candidate(f"Under {linea_casino}", prob_ml_under, prob_mc_under, cuotas_reales['Cuota_Under'], mkt_under, carreras_dict.get(f"Push {linea_casino}", 0.0))''',
'''runline_candidate(f"Hándicap {spread_loc:+.1f} ({datos_partido['local']})", prob_ml_spread_loc, prob_mc_spread_loc, cuotas_reales['Cuota_Spread_Local'], mkt_sp_loc)''':
'''runline_candidate(f"Hándicap {spread_loc:+.1f} ({datos_partido['local']})", prob_ml_spread_loc, prob_mc_spread_loc, cuotas_reales['Cuota_Spread_Local'], mkt_sp_loc, carreras_dict.get(f"Push Spread Local {spread_loc:+.1f}", 0.0))''',
'''runline_candidate(f"Hándicap {spread_vis:+.1f} ({datos_partido['visita']})", prob_ml_spread_vis, prob_mc_spread_vis, cuotas_reales['Cuota_Spread_Visita'], mkt_sp_vis)''':
'''runline_candidate(f"Hándicap {spread_vis:+.1f} ({datos_partido['visita']})", prob_ml_spread_vis, prob_mc_spread_vis, cuotas_reales['Cuota_Spread_Visita'], mkt_sp_vis, carreras_dict.get(f"Push Spread Visita {spread_vis:+.1f}", 0.0))''',
}
for old, new in repls.items():
    t = must_replace(t, old, new, 'push-aware candidate', 1)

# Kelly with pushes.
kelly_pattern = re.compile(r'def calcular_criterio_kelly\(.*?\n(?=def estimar_prob_ml)', re.S)
km = kelly_pattern.search(t)
if not km:
    raise SystemExit('Kelly block not found')
new_kelly = '''def calcular_criterio_kelly(probabilidad_real, cuota_decimal, fraccion=0.25, prob_push=0.0):
    """Fractional Kelly with push/refund probability handled explicitly."""
    try:
        if cuota_decimal is None or probabilidad_real is None:
            return 0.0
        p = max(0.0, float(probabilidad_real) / 100.0)
        push = max(0.0, float(prob_push) / 100.0)
        q = max(0.0, 1.0 - p - push)
        b = float(cuota_decimal) - 1.0
        decisions = p + q
        if b <= 0 or decisions <= 0:
            return 0.0
        kelly = (b * p - q) / (b * decisions)
        return round(max(0.0, kelly * fraccion) * 100.0, 2)
    except Exception:
        return 0.0

'''
t = t[:km.start()] + new_kelly + t[km.end():]

t = t.replace('calcular_criterio_kelly(cand.probability, cand.odds)', 'calcular_criterio_kelly(cand.probability, cand.odds, prob_push=cand.push_probability)')

t = must_replace(t,
'''                            'Prob. Combinada': f"{round(cand.probability, 1)}%",
                            'Cuota': cand.odds,
''',
'''                            'Prob. Combinada': f"{round(cand.probability, 1)}%",
                            'Push': f"{round(cand.push_probability, 1)}%" if cand.push_probability else "0.0%",
                            'Cuota': cand.odds,
''',
'individual push display', 1)

# Surface scanner data failures rather than silently pretending NO BET.
t = t.replace('                        except Exception as e:\n                            continue\n',
'''                        except Exception as e:
                            errores_datos.append({"Partido": f"{datos_partido.get('visita','?')} @ {datos_partido.get('local','?')}", "Error": str(e)[:180]})
                            continue
''', 1)

marker = '''                    if diagnostico_totales:
                        st.markdown("### 🧪 Mejor oportunidad O/U analizada")
                        df_tot_diag=pd.DataFrame(diagnostico_totales)
                        df_tot_diag["_rank"]=df_tot_diag["Edge_pp"].fillna(-999)+df_tot_diag["EV_pct"].fillna(-999)
                        st.dataframe(df_tot_diag.sort_values("_rank",ascending=False).head(3).drop(columns=["_rank"]), use_container_width=True, hide_index=True)
'''
replacement = marker + '''
                    if errores_datos:
                        with st.expander(f"⚠️ Partidos no evaluados por datos incompletos ({len(errores_datos)})"):
                            st.dataframe(pd.DataFrame(errores_datos), use_container_width=True, hide_index=True)
'''
t = must_replace(t, marker, replacement, 'scanner diagnostics UI', 1)

# Warn clearly when production has no private odds key.
marker2 = '        modo_app = st.sidebar.radio("Modo de Operación", ["🎯 Análisis Individual por Partido", "🔍 Escáner Automático de la Jornada (EV+)"])\n'
replacement2 = '''        if not ODDS_API_KEY:
            st.warning("⚠️ ODDS_API_KEY no configurada en Secrets/entorno. La cartelera se muestra, pero no se emitirán apuestas sin cuotas reales.")
        if not persistent_backend_available():
            st.caption("ℹ️ Ledger en modo local: configura GITHUB_TOKEN de escritura para persistencia entre reinicios.")
        modo_app = st.sidebar.radio("Modo de Operación", ["🎯 Análisis Individual por Partido", "🔍 Escáner Automático de la Jornada (EV+)"])
'''
t = must_replace(t, marker2, replacement2, 'runtime warnings', 1)

p.write_text(t, encoding='utf-8')

# ---------------- modules/ml_mlb.py ----------------
p = Path('modules/ml_mlb.py')
t = p.read_text(encoding='utf-8')
t = must_replace(t, 'import math\nimport threading\n', 'import hashlib\nimport math\nimport threading\n', 'ml hashlib', 1)

sig_pattern = re.compile(r'def _frame_signature\(.*?\n\ndef _cache_key', re.S)
sm = sig_pattern.search(t)
if not sm:
    raise SystemExit('frame signature block not found')
new_sig = '''def _frame_signature(df, important_columns):
    """Content-sensitive signature so changed middle rows cannot reuse stale estimators."""
    if df is None or df.empty:
        return (0, 0, "")
    cols = [c for c in important_columns if c in df.columns]
    if not cols:
        cols = list(df.columns)
    hashed = pd.util.hash_pandas_object(df[cols], index=True).values.tobytes()
    digest = hashlib.sha256(hashed).hexdigest()
    return (int(len(df)), int(len(df.columns)), digest)


def _cache_key'''
t = t[:sm.start()] + new_sig + t[sm.end():]
p.write_text(t, encoding='utf-8')

print('V5 deterministic app/ML patch applied')
