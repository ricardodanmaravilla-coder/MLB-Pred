from pathlib import Path

p = Path('app_mlb.py')
s = p.read_text(encoding='utf-8')
orig = s

# Add a market-relative ranking score next to the existing value filter.
needle = '''def calcular_criterio_kelly(probabilidad_real, cuota_decimal, fraccion=0.25):\n'''
helper = '''def _score_valor(prob_pct, cuota, mercado_no_vig=None, desacuerdo_pp=0.0):\n    """Rank candidates by market-relative value, not raw probability.\n\n    This prevents naturally high-base-rate markets such as +1.5 from\n    dominating merely because their nominal win probability is larger.\n    """\n    try:\n        p = float(prob_pct) / 100.0\n        o = float(cuota)\n        ev_pct = (p * o - 1.0) * 100.0\n        edge_pp = 0.0 if mercado_no_vig is None else (p - float(mercado_no_vig)) * 100.0\n        disagreement_penalty = max(0.0, float(desacuerdo_pp)) * 0.15\n        return round((1.5 * edge_pp) + ev_pct - disagreement_penalty, 4)\n    except (TypeError, ValueError):\n        return -999.0\n\n\n'''
if '_score_valor(' not in s:
    s = s.replace(needle, helper + needle, 1)

# Totals should no longer require an almost-unreachable 59% from each engine.
# They still need both engines on the same side, combined confidence, bounded
# disagreement, and stronger market-relative edge/EV.
s = s.replace('umbral_totales = 59.0  # Totales', 'umbral_totales = 52.0  # Totales: consenso + valor relativo al mercado')

old_over = '''                            # 3. Totales Over\n                            cuota_ov = datos_partido.get("cuota_over")\n                            if cuota_ov is not None and prob_mc_over >= umbral_totales and prob_ml_over >= umbral_totales:\n                                prob_comb_over = (prob_mc_over + prob_ml_over) / 2.0\n                                ev_over = (prob_comb_over / 100.0) * cuota_ov - 1.0\n                                if _pasa_valor(prob_comb_over, cuota_ov, mkt_over_scanner):\n                                    recomendaciones.append({\n'''
new_over = '''                            # 3. Totales Over: consenso + edge/EV, no un 59% absoluto casi inalcanzable.\n                            cuota_ov = datos_partido.get("cuota_over")\n                            if cuota_ov is not None:\n                                prob_comb_over = (prob_mc_over + prob_ml_over) / 2.0\n                                desac_over = abs(prob_mc_over - prob_ml_over)\n                                ev_over = (prob_comb_over / 100.0) * cuota_ov - 1.0\n                                if (prob_mc_over >= umbral_totales and prob_ml_over >= umbral_totales and\n                                    prob_comb_over >= 55.0 and desac_over <= 10.0 and\n                                    _pasa_valor(prob_comb_over, cuota_ov, mkt_over_scanner, min_ev=0.04, min_edge=0.04)):\n                                    recomendaciones.append({\n'''
s = s.replace(old_over, new_over, 1)

old_under = '''                            # 4. Totales Under\n                            cuota_un = datos_partido.get("cuota_under")\n                            if cuota_un is not None and prob_mc_under >= umbral_totales and prob_ml_under >= umbral_totales:\n                                prob_comb_under = (prob_mc_under + prob_ml_under) / 2.0\n                                ev_under = (prob_comb_under / 100.0) * cuota_un - 1.0\n                                if _pasa_valor(prob_comb_under, cuota_un, mkt_under_scanner):\n                                    recomendaciones.append({\n'''
new_under = '''                            # 4. Totales Under: mismo estándar que Over.\n                            cuota_un = datos_partido.get("cuota_under")\n                            if cuota_un is not None:\n                                prob_comb_under = (prob_mc_under + prob_ml_under) / 2.0\n                                desac_under = abs(prob_mc_under - prob_ml_under)\n                                ev_under = (prob_comb_under / 100.0) * cuota_un - 1.0\n                                if (prob_mc_under >= umbral_totales and prob_ml_under >= umbral_totales and\n                                    prob_comb_under >= 55.0 and desac_under <= 10.0 and\n                                    _pasa_valor(prob_comb_under, cuota_un, mkt_under_scanner, min_ev=0.04, min_edge=0.04)):\n                                    recomendaciones.append({\n'''
s = s.replace(old_under, new_under, 1)

# Make run-line earn its place with stronger market-relative value.
s = s.replace('_pasa_valor(prob_comb_sp_loc, cuota_sp_loc, mkt_sp_loc_scanner)):', '_pasa_valor(prob_comb_sp_loc, cuota_sp_loc, mkt_sp_loc_scanner, min_ev=0.04, min_edge=0.04)):', 1)
s = s.replace('_pasa_valor(prob_comb_sp_vis, cuota_sp_vis, mkt_sp_vis_scanner)):', '_pasa_valor(prob_comb_sp_vis, cuota_sp_vis, mkt_sp_vis_scanner, min_ev=0.04, min_edge=0.04)):', 1)

# Add hidden ranking score to every accepted candidate. We rank by value against
# the market, then show only the best three without forcing a market quota.
repls = [
    ('"Stake Kelly": f"{calcular_criterio_kelly(prob_comb_loc, cuota_loc)}%"', '"Stake Kelly": f"{calcular_criterio_kelly(prob_comb_loc, cuota_loc)}%",\n                                        "_Score": _score_valor(prob_comb_loc, cuota_loc, mkt_loc_scanner, abs(prob_mc_loc-prob_ml_loc))'),
    ('"Stake Kelly": f"{calcular_criterio_kelly(prob_comb_vis, cuota_vis)}%"', '"Stake Kelly": f"{calcular_criterio_kelly(prob_comb_vis, cuota_vis)}%",\n                                        "_Score": _score_valor(prob_comb_vis, cuota_vis, mkt_vis_scanner, abs(prob_mc_vis-prob_ml_vis))'),
    ('"Stake Kelly": f"{calcular_criterio_kelly(prob_comb_over, cuota_ov)}%"', '"Stake Kelly": f"{calcular_criterio_kelly(prob_comb_over, cuota_ov)}%",\n                                        "_Score": _score_valor(prob_comb_over, cuota_ov, mkt_over_scanner, desac_over)'),
    ('"Stake Kelly": f"{calcular_criterio_kelly(prob_comb_under, cuota_un)}%"', '"Stake Kelly": f"{calcular_criterio_kelly(prob_comb_under, cuota_un)}%",\n                                        "_Score": _score_valor(prob_comb_under, cuota_un, mkt_under_scanner, desac_under)'),
    ('"Stake Kelly": f"{calcular_criterio_kelly(prob_comb_sp_loc, cuota_sp_loc)}%"', '"Stake Kelly": f"{calcular_criterio_kelly(prob_comb_sp_loc, cuota_sp_loc)}%",\n                                        "_Score": _score_valor(prob_comb_sp_loc, cuota_sp_loc, mkt_sp_loc_scanner, desac_sp_loc)'),
    ('"Stake Kelly": f"{calcular_criterio_kelly(prob_comb_sp_vis, cuota_sp_vis)}%"', '"Stake Kelly": f"{calcular_criterio_kelly(prob_comb_sp_vis, cuota_sp_vis)}%",\n                                        "_Score": _score_valor(prob_comb_sp_vis, cuota_sp_vis, mkt_sp_vis_scanner, desac_sp_vis)'),
]
for old, new in repls:
    s = s.replace(old, new, 1)

old_display = '''                    if recomendaciones:\n                        df_recom = pd.DataFrame(recomendaciones)\n                        st.dataframe(df_recom, use_container_width=True, hide_index=True)\n'''
new_display = '''                    if recomendaciones:\n                        df_recom = pd.DataFrame(recomendaciones)\n                        if "_Score" in df_recom.columns:\n                            df_recom = df_recom.sort_values("_Score", ascending=False).head(3).drop(columns=["_Score"])\n                        st.dataframe(df_recom, use_container_width=True, hide_index=True)\n'''
s = s.replace(old_display, new_display, 1)

if s == orig:
    raise SystemExit('Patch made no changes')
required = [
    'def _score_valor',
    'prob_comb_over >= 55.0',
    'prob_comb_under >= 55.0',
    'min_ev=0.04, min_edge=0.04',
    'sort_values("_Score", ascending=False).head(3)',
]
missing = [x for x in required if x not in s]
if missing:
    raise SystemExit(f'Missing expected markers: {missing}')
p.write_text(s, encoding='utf-8')
print('market ranking patch applied')
