import pandas as pd

def analizar_apuestas_mlb(res_montecarlo, preds_ml, cuotas):
    """
    Evalúa Value Odds (EV+) en los 4 mercados principales con filtro Sniper (>60% en ambos).
    """
    filas = []
    
    # 1. Mercado Moneyline Local
    p_mc_loc = res_montecarlo["Moneyline"]["Gana Local"]
    p_ml_loc = preds_ml.get("Prob_Local_ML", p_mc_loc)
    cuota_loc = cuotas.get("Moneyline_Local", 1.90)
    
    if cuota_loc > 1.0:
        p_consenso = (p_mc_loc + p_ml_loc) / 2.0
        ev = (p_consenso / 100.0 * cuota_loc) - 1.0
        
        # Filtro Sniper: Ambos modelos deben superar el 60%
        es_sniper = (p_mc_loc >= 60.0) and (p_ml_loc >= 60.0) and (ev > 0)
        
        veredicto = "❌ Sin Valor"
        if es_sniper:
            veredicto = "🔥 APUESTA FRANCOTIRADOR (EV+)"
        elif ev > 0.05:
            veredicto = "✅ Valor Positivo (EV+)"

        filas.append({
            "Mercado": "Gana Local (Moneyline)",
            "Montecarlo (%)": f"{p_mc_loc}%",
            "ML (%)": f"{p_ml_loc}%",
            "Cuota Bookie": f"{cuota_loc:.2f}",
            "EV (%)": f"{ev * 100:.1f}%",
            "Veredicto": veredicto
        })

    # 2. Mercado Over Carreras
    p_mc_over = res_montecarlo["Carreras"]["Over 8.5"]
    cuota_over = cuotas.get("Over_8.5_Carreras", 1.90)
    if cuota_over > 1.0:
        ev_over = (p_mc_over / 100.0 * cuota_over) - 1.0
        veredicto_over = "🔥 APUESTA FRANCOTIRADOR" if (p_mc_over >= 60.0 and ev_over > 0) else ("✅ EV+" if ev_over > 0 else "❌ Sin Valor")
        
        filas.append({
            "Mercado": "Over 8.5 Carreras",
            "Montecarlo (%)": f"{p_mc_over}%",
            "ML (%)": f"{preds_ml.get('Carreras_Proyectadas_ML', 8.5)} proyectadas",
            "Cuota Bookie": f"{cuota_over:.2f}",
            "EV (%)": f"{ev_over * 100:.1f}%",
            "Veredicto": veredicto_over
        })

    return pd.DataFrame(filas)
