import pandas as pd

def analizar_apuestas_mlb(res_montecarlo, preds_ml, cuotas_reales, linea_carreras):
    """
    Evalúa Value Odds (EV+) usando las cuotas reales del casino obtenidas de ESPN.
    """
    filas = []
    
    # 1. Mercado Moneyline Local
    p_mc_loc = res_montecarlo["Moneyline"]["Gana Local"]
    p_ml_loc = preds_ml.get("Prob_Local_ML", p_mc_loc)
    cuota_loc = cuotas_reales.get("Moneyline_Local", 0.0)
    
    if cuota_loc > 1.0:
        p_consenso = (p_mc_loc + p_ml_loc) / 2.0
        ev = (p_consenso / 100.0 * cuota_loc) - 1.0
        
        es_sniper = (p_mc_loc >= 60.0) and (p_ml_loc >= 60.0) and (ev > 0)
        veredicto = "🔥 APUESTA FRANCOTIRADOR" if es_sniper else ("✅ EV+" if ev > 0.03 else "❌ Sin Valor")

        filas.append({
            "Mercado": "Gana Local (Moneyline)",
            "Montecarlo (%)": f"{p_mc_loc}%",
            "ML (%)": f"{p_ml_loc}%",
            "Cuota Casino": f"{cuota_loc:.2f}",
            "EV (%)": f"{ev * 100:.1f}%",
            "Veredicto": veredicto
        })

    # 2. Mercado Totales Dinámico (Over)
    llave_over = f"Over {linea_carreras}"
    p_mc_over = res_montecarlo["Carreras"].get(llave_over, 0)
    cuota_over = cuotas_reales.get("Cuota_Over", 0.0)
    
    if cuota_over > 1.0:
        ev_over = (p_mc_over / 100.0 * cuota_over) - 1.0
        veredicto_over = "🔥 APUESTA FRANCOTIRADOR" if (p_mc_over >= 60.0 and ev_over > 0) else ("✅ EV+" if ev_over > 0 else "❌ Sin Valor")
        
        filas.append({
            "Mercado": llave_over,
            "Montecarlo (%)": f"{p_mc_over}%",
            "ML (%)": f"{preds_ml.get('Carreras_Proyectadas_ML', 0)} proyectadas",
            "Cuota Casino": f"{cuota_over:.2f}",
            "EV (%)": f"{ev_over * 100:.1f}%",
            "Veredicto": veredicto_over
        })

    return pd.DataFrame(filas)
