from pathlib import Path

p = Path('app_mlb.py')
s = p.read_text(encoding='utf-8')
orig = s

# 1) Allow historically calibrated residual sigma instead of hard-coded values.
s = s.replace('def estimar_prob_ml(proyeccion, linea, tipo="over"):', 'def estimar_prob_ml(proyeccion, linea, tipo="over", sigma=None):')
s = s.replace('''    # Desviación estándar (RMSE) estimada empíricamente para MLB. \n    # (Si mides el error real de tu modelo, ajusta estos valores)\n    sigma_carreras = 2.45  # El error promedio prediciendo Totales\n    sigma_spread = 1.75    # El error promedio prediciendo el Hándicap (Diferencial)\n    \n    try:\n        if tipo == "over":\n            z = (proyeccion - linea) / sigma_carreras\n        elif tipo == "under":\n            z = (linea - proyeccion) / sigma_carreras\n        elif tipo in ["spread_loc", "spread_vis"]:\n            z = (proyeccion + linea) / sigma_spread \n''', '''    # Sigma comes from chronological out-of-sample residuals when available.\n    try:\n        sigma = float(sigma) if sigma is not None else (3.5 if tipo in ["over", "under"] else 4.2)\n        sigma = max(1.0, sigma)\n        if tipo == "over":\n            z = (proyeccion - linea) / sigma\n        elif tipo == "under":\n            z = (linea - proyeccion) / sigma\n        elif tipo in ["spread_loc", "spread_vis"]:\n            z = (proyeccion + linea) / sigma\n''')

# 2) Use real cached weather in scanner, with conservative fallback only if lookup fails.
s = s.replace('''                            # Ejecutar Simulación de Montecarlo\n                            res_mc = simular_partido_mlb(\n''', '''                            # Clima real del estadio para el scanner; fallback solo si la consulta no responde.\n                            temp_scan, viento_scan, dir_scan = obtener_clima_estadio(datos_partido["local"])\n                            temp_scan = 72 if temp_scan is None else temp_scan\n                            viento_scan = 8 if viento_scan is None else viento_scan\n                            dir_scan = "None" if not dir_scan else dir_scan\n\n                            # Ejecutar Simulación de Montecarlo\n                            res_mc = simular_partido_mlb(\n''', 1)
s = s.replace('''                                viento_mph=8, direccion_viento="None", temp_f=72,\n''', '''                                viento_mph=viento_scan, direccion_viento=dir_scan, temp_f=temp_scan,\n''', 1)

# 3) Calibrated ML probabilities in scanner.
s = s.replace('prob_ml_over = estimar_prob_ml(proy_carreras, linea_casino, "over")', 'prob_ml_over = estimar_prob_ml(proy_carreras, linea_casino, "over", res_ml.get("Sigma_Carreras"))')
s = s.replace('prob_ml_under = estimar_prob_ml(proy_carreras, linea_casino, "under")', 'prob_ml_under = estimar_prob_ml(proy_carreras, linea_casino, "under", res_ml.get("Sigma_Carreras"))')
s = s.replace('prob_ml_spread_loc = estimar_prob_ml(proy_hc_loc, spread_loc, "spread_loc") if spread_loc is not None else 50.0', 'prob_ml_spread_loc = estimar_prob_ml(proy_hc_loc, spread_loc, "spread_loc", res_ml.get("Sigma_Handicap")) if spread_loc is not None else 50.0')
s = s.replace('prob_ml_spread_vis = estimar_prob_ml(-proy_hc_loc, spread_vis, "spread_vis") if spread_vis is not None else 50.0', 'prob_ml_spread_vis = estimar_prob_ml(-proy_hc_loc, spread_vis, "spread_vis", res_ml.get("Sigma_Handicap")) if spread_vis is not None else 50.0')

# 4) Run-line must be supported by both engines, have bounded disagreement, edge and EV.
old_loc = '''                            # 5. Spread Local (Lectura dinámica de la matriz de Montecarlo)\n                            cuota_sp_loc = datos_partido.get("cuota_spread_loc")\n                            if spread_loc is not None and cuota_sp_loc is not None:\n                                ev_sp_loc = (prob_mc_spread_loc / 100.0) * cuota_sp_loc - 1.0\n                                \n                                if prob_mc_spread_loc >= umbral_handicap and _pasa_valor(prob_mc_spread_loc, cuota_sp_loc, mkt_sp_loc_scanner):\n                                    recomendaciones.append({\n'''
new_loc = '''                            # 5. Spread Local: consenso MC + ML; evita sesgo sistemático hacia +1.5.\n                            cuota_sp_loc = datos_partido.get("cuota_spread_loc")\n                            if spread_loc is not None and cuota_sp_loc is not None:\n                                prob_comb_sp_loc = (prob_mc_spread_loc + prob_ml_spread_loc) / 2.0\n                                desac_sp_loc = abs(prob_mc_spread_loc - prob_ml_spread_loc)\n                                ev_sp_loc = (prob_comb_sp_loc / 100.0) * cuota_sp_loc - 1.0\n                                \n                                if (prob_mc_spread_loc >= 56.0 and prob_ml_spread_loc >= 54.0 and\n                                    prob_comb_sp_loc >= 56.0 and desac_sp_loc <= 12.0 and\n                                    _pasa_valor(prob_comb_sp_loc, cuota_sp_loc, mkt_sp_loc_scanner)):\n                                    recomendaciones.append({\n'''
s = s.replace(old_loc, new_loc)
s = s.replace('"Stake Kelly": f"{calcular_criterio_kelly(prob_mc_spread_loc, cuota_sp_loc)}%"', '"Stake Kelly": f"{calcular_criterio_kelly(prob_comb_sp_loc, cuota_sp_loc)}%"', 1)

old_vis = '''                            # 6. Spread Visita\n                            cuota_sp_vis = datos_partido.get("cuota_spread_vis")\n                            if spread_vis is not None and cuota_sp_vis is not None:\n                                ev_sp_vis = (prob_mc_spread_vis / 100.0) * cuota_sp_vis - 1.0\n                                \n                                if prob_mc_spread_vis >= umbral_handicap and _pasa_valor(prob_mc_spread_vis, cuota_sp_vis, mkt_sp_vis_scanner):\n                                    recomendaciones.append({\n'''
new_vis = '''                            # 6. Spread Visita: mismo estándar de consenso que el local.\n                            cuota_sp_vis = datos_partido.get("cuota_spread_vis")\n                            if spread_vis is not None and cuota_sp_vis is not None:\n                                prob_comb_sp_vis = (prob_mc_spread_vis + prob_ml_spread_vis) / 2.0\n                                desac_sp_vis = abs(prob_mc_spread_vis - prob_ml_spread_vis)\n                                ev_sp_vis = (prob_comb_sp_vis / 100.0) * cuota_sp_vis - 1.0\n                                \n                                if (prob_mc_spread_vis >= 56.0 and prob_ml_spread_vis >= 54.0 and\n                                    prob_comb_sp_vis >= 56.0 and desac_sp_vis <= 12.0 and\n                                    _pasa_valor(prob_comb_sp_vis, cuota_sp_vis, mkt_sp_vis_scanner)):\n                                    recomendaciones.append({\n'''
s = s.replace(old_vis, new_vis)
s = s.replace('"Stake Kelly": f"{calcular_criterio_kelly(prob_mc_spread_vis, cuota_sp_vis)}%"', '"Stake Kelly": f"{calcular_criterio_kelly(prob_comb_sp_vis, cuota_sp_vis)}%"', 1)

# 5) Eliminate invented fallback probability; if a requested spread key is absent, neutral/no-pick.
s = s.replace('carreras_dict.get(f"Spread Local {spread_loc:+.1f}", prob_mc_loc * 0.90)', 'carreras_dict.get(f"Spread Local {spread_loc:+.1f}", 50.0)')
s = s.replace('carreras_dict.get(f"Spread Visita {spread_vis:+.1f}", prob_mc_vis * 0.90)', 'carreras_dict.get(f"Spread Visita {spread_vis:+.1f}", 50.0)')

if s == orig:
    raise SystemExit('Patch made no changes')
required = [
    'sigma=None', 'viento_mph=viento_scan', 'prob_comb_sp_loc', 'prob_comb_sp_vis',
    'res_ml.get("Sigma_Carreras")', 'res_ml.get("Sigma_Handicap")'
]
missing = [x for x in required if x not in s]
if missing:
    raise SystemExit(f'Missing expected patch markers: {missing}')
p.write_text(s, encoding='utf-8')
print('runline/weather/calibration patch applied')
