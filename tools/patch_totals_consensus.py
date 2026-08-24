from pathlib import Path

p=Path('app_mlb.py')
s=p.read_text(encoding='utf-8')
orig=s

# 1) Keep ML inference on the same feature type it saw in training: team pitching ERA.
s=s.replace(
    "res_ml = predictor_ml.predecir_partido(loc_abbr, vis_abbr, wrc_loc, wrc_vis, xfip_loc, xfip_vis, park_factor)",
    "res_ml = predictor_ml.predecir_partido(loc_abbr, vis_abbr, wrc_loc, wrc_vis, bullpen_loc_era, bullpen_vis_era, park_factor)  # ML trained on team pitching, not starter ERA",
    1,
)

# 2) Evidence from chronological O/U diagnostics supports evaluating around 54%,
# while edge/EV and ML/MC agreement remain strict.
s=s.replace('prob_comb_over >= 55.0 and desac_over <= 10.0', 'prob_comb_over >= 54.0 and desac_over <= 10.0', 1)
s=s.replace('prob_comb_under >= 55.0 and desac_under <= 10.0', 'prob_comb_under >= 54.0 and desac_under <= 10.0', 1)

# 3) Add a compact diagnostic helper so a missing O/U recommendation is explainable.
needle='''def calcular_criterio_kelly(probabilidad_real, cuota_decimal, fraccion=0.25):\n'''
helper='''def _diagnostico_total(prob_ml, prob_mc, prob_comb, cuota, mercado_no_vig, desacuerdo):\n    try:\n        p=float(prob_comb)/100.0\n        ev=(p*float(cuota)-1.0)*100.0 if cuota is not None else None\n        edge=(p-float(mercado_no_vig))*100.0 if mercado_no_vig is not None else None\n        checks=[\n            (float(prob_ml)>=52.0, f"ML {float(prob_ml):.1f}% < 52%"),\n            (float(prob_mc)>=52.0, f"MC {float(prob_mc):.1f}% < 52%"),\n            (float(prob_comb)>=54.0, f"Combinada {float(prob_comb):.1f}% < 54%"),\n            (float(desacuerdo)<=10.0, f"Desacuerdo {float(desacuerdo):.1f} pp > 10"),\n            (edge is not None and edge>=4.0, "Edge < 4 pp"),\n            (ev is not None and ev>=4.0, "EV < 4%"),\n        ]\n        fails=[msg for ok,msg in checks if not ok]\n        return {"EV_pct": None if ev is None else round(ev,2), "Edge_pp": None if edge is None else round(edge,2), "Estado": "CANDIDATO" if not fails else "NO BET", "Motivo": "Cumple filtros O/U" if not fails else "; ".join(fails)}\n    except Exception as e:\n        return {"EV_pct":None,"Edge_pp":None,"Estado":"NO BET","Motivo":f"Error diagnóstico: {e}"}\n\n\n'''
if 'def _diagnostico_total(' not in s:
    s=s.replace(needle,helper+needle,1)

s=s.replace('''                    recomendaciones = []\n                    \n                    for llave, datos_partido in partidos_hoy.items():''','''                    recomendaciones = []\n                    diagnostico_totales = []\n                    \n                    for llave, datos_partido in partidos_hoy.items():''',1)

# Add O/U diagnostics after no-vig probabilities are available and before filtering.
needle2='''                            mkt_sp_loc_scanner, mkt_sp_vis_scanner = _prob_no_vig_dos_vias(\n                                datos_partido.get("cuota_spread_loc"), datos_partido.get("cuota_spread_vis")\n                            )\n\n                            \n                            # 1. Moneyline Local\n'''
insert2='''                            mkt_sp_loc_scanner, mkt_sp_vis_scanner = _prob_no_vig_dos_vias(\n                                datos_partido.get("cuota_spread_loc"), datos_partido.get("cuota_spread_vis")\n                            )\n\n                            # Diagnóstico O/U: se registra aun cuando no llega al top 3.\n                            for lado, pml_t, pmc_t, cuota_t, mkt_t in [\n                                (f"Over {linea_casino}", prob_ml_over, prob_mc_over, datos_partido.get("cuota_over"), mkt_over_scanner),\n                                (f"Under {linea_casino}", prob_ml_under, prob_mc_under, datos_partido.get("cuota_under"), mkt_under_scanner),\n                            ]:\n                                if cuota_t is not None:\n                                    pcomb_t=(pml_t+pmc_t)/2.0\n                                    desac_t=abs(pml_t-pmc_t)\n                                    dg=_diagnostico_total(pml_t,pmc_t,pcomb_t,cuota_t,mkt_t,desac_t)\n                                    diagnostico_totales.append({\n                                        "Partido": f"{datos_partido['visita']} @ {datos_partido['local']}",\n                                        "O/U": lado, "ML": round(pml_t,1), "MC": round(pmc_t,1),\n                                        "Combinada": round(pcomb_t,1), "Cuota": cuota_t,\n                                        "Edge_pp": dg["Edge_pp"], "EV_pct": dg["EV_pct"],\n                                        "Estado": dg["Estado"], "Motivo": dg["Motivo"],\n                                    })\n\n                            # 1. Moneyline Local\n'''
s=s.replace(needle2,insert2,1)

# Show the best O/U diagnostic after the normal recommendation table/info.
old='''                    if recomendaciones:\n                        df_recom = pd.DataFrame(recomendaciones)\n                        if "_Score" in df_recom.columns:\n                            df_recom = df_recom.sort_values("_Score", ascending=False).head(3).drop(columns=["_Score"])\n                        st.dataframe(df_recom, use_container_width=True, hide_index=True)\n                    else:\n                        st.info("No se encontraron partidos con EV+ y más del 60% de probabilidad en ambos modelos hoy.")\n'''
new='''                    if recomendaciones:\n                        df_recom = pd.DataFrame(recomendaciones)\n                        if "_Score" in df_recom.columns:\n                            df_recom = df_recom.sort_values("_Score", ascending=False).head(3).drop(columns=["_Score"])\n                        st.dataframe(df_recom, use_container_width=True, hide_index=True)\n                    else:\n                        st.info("No se encontraron apuestas que superen los filtros de valor del scanner hoy.")\n\n                    if diagnostico_totales:\n                        st.markdown("### 🧪 Mejor oportunidad O/U analizada")\n                        df_tot_diag=pd.DataFrame(diagnostico_totales)\n                        df_tot_diag["_rank"]=df_tot_diag["Edge_pp"].fillna(-999)+df_tot_diag["EV_pct"].fillna(-999)\n                        st.dataframe(df_tot_diag.sort_values("_rank",ascending=False).head(3).drop(columns=["_rank"]), use_container_width=True, hide_index=True)\n'''
s=s.replace(old,new,1)

if s==orig:
    raise SystemExit('No changes applied')
required=['ML trained on team pitching','prob_comb_over >= 54.0','prob_comb_under >= 54.0','def _diagnostico_total','Mejor oportunidad O/U analizada']
missing=[x for x in required if x not in s]
if missing:
    raise SystemExit(f'Missing markers: {missing}')
p.write_text(s,encoding='utf-8')
print('totals consensus patch applied')
