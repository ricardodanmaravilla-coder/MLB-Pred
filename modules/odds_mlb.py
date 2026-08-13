import pandas as pd

def analizar_apuestas_mlb(res_mc, preds_ml, cuotas_reales, linea_carreras):
    """
    Analiza el valor esperado (EV+) comparando las probabilidades de Montecarlo 
    contra las cuotas reales del casino para Local, Visita, Over y Under dinámicos.
    """
    apuestas = []
    
    # 1. Probabilidades de Montecarlo
    prob_loc = res_mc["Moneyline"]["Gana Local"] / 100.0
    prob_vis = res_mc["Moneyline"]["Gana Visita"] / 100.0
    
    clave_over = f"Over {linea_carreras}"
    clave_under = f"Under {linea_carreras}"
    
    prob_over = res_mc["Carreras"].get(clave_over, 50.0) / 100.0
    prob_under = res_mc["Carreras"].get(clave_under, 50.0) / 100.0
    
    # Cuotas del casino sin datos inventados
    cuota_loc = cuotas_reales.get("Moneyline_Local")
    cuota_vis = cuotas_reales.get("Moneyline_Visita") 
    cuota_over = cuotas_reales.get("Cuota_Over")
    cuota_under = cuotas_reales.get("Cuota_Under") 
    
    # --- EVALUACIÓN DE VALOR (EV+) ---
    if cuota_loc is not None:
        ev_loc = (prob_loc * cuota_loc) - 1
        apuestas.append({
            "Seleccion": "Moneyline Local",
            "Prob Model": f"{round(prob_loc * 100, 1)}%",
            "Cuota": cuota_loc,
            "EV+": f"{round(ev_loc * 100, 2)}%",
            "Veredicto": "🔥 ¡Apostar!" if ev_loc > 0.03 else ("✅ Valor Justo" if ev_loc > 0 else "❌ Sin Valor")
        })

    if cuota_vis is not None:
        ev_vis = (prob_vis * cuota_vis) - 1
        apuestas.append({
            "Seleccion": "Moneyline Visitante",
            "Prob Model": f"{round(prob_vis * 100, 1)}%",
            "Cuota": cuota_vis,
            "EV+": f"{round(ev_vis * 100, 2)}%",
            "Veredicto": "🔥 ¡Apostar!" if ev_vis > 0.03 else ("✅ Valor Justo" if ev_vis > 0 else "❌ Sin Valor")
        })

    if cuota_over is not None:
        ev_over = (prob_over * cuota_over) - 1
        apuestas.append({
            "Seleccion": f"Over {linea_carreras} Carreras",
            "Prob Model": f"{round(prob_over * 100, 1)}%",
            "Cuota": cuota_over,
            "EV+": f"{round(ev_over * 100, 2)}%",
            "Veredicto": "🔥 ¡Apostar!" if ev_over > 0.03 else ("✅ Valor Justo" if ev_over > 0 else "❌ Sin Valor")
        })

    if cuota_under is not None:
        ev_under = (prob_under * cuota_under) - 1
        apuestas.append({
            "Seleccion": f"Under {linea_carreras} Carreras",
            "Prob Model": f"{round(prob_under * 100, 1)}%",
            "Cuota": cuota_under,
            "EV+": f"{round(ev_under * 100, 2)}%",
            "Veredicto": "🔥 ¡Apostar!" if ev_under > 0.03 else ("✅ Valor Justo" if ev_under > 0 else "❌ Sin Valor")
        })

    return pd.DataFrame(apuestas)
