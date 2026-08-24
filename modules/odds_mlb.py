import math
import pandas as pd


def american_to_decimal(value):
    if value is None:
        return None
    x = float(value)
    if x == 0:
        return None
    return 1.0 + (x / 100.0 if x > 0 else 100.0 / abs(x))


def remove_vig_two_way(odd_a, odd_b):
    ia, ib = 1.0 / float(odd_a), 1.0 / float(odd_b)
    s = ia + ib
    return ia / s * 100.0, ib / s * 100.0


def expected_value(prob_pct, odd, push_pct=0.0):
    p = float(prob_pct) / 100.0
    push = max(0.0, float(push_pct)) / 100.0
    loss = max(0.0, 1.0 - p - push)
    return (p * (float(odd) - 1.0) - loss) * 100.0


def market_edge(prob_pct, market_prob_pct, push_pct=0.0):
    decisive = max(1e-9, 100.0 - float(push_pct))
    conditional = float(prob_pct) / decisive * 100.0
    return conditional - float(market_prob_pct), conditional


def evaluar_dos_vias(nombre_a, nombre_b, p_a, p_b, odd_a, odd_b, push_a=0.0, push_b=0.0,
                      min_edge=3.0, min_ev=2.0, min_prob=52.0):
    if not odd_a or not odd_b:
        return []
    market_a, market_b = remove_vig_two_way(odd_a, odd_b)
    rows = []
    for name, prob, odd, market_p, push in (
        (nombre_a, p_a, odd_a, market_a, push_a),
        (nombre_b, p_b, odd_b, market_b, push_b),
    ):
        edge, cond = market_edge(prob, market_p, push)
        ev = expected_value(prob, odd, push)
        reasons = []
        if cond < min_prob: reasons.append(f"prob {cond:.1f}<{min_prob:.1f}%")
        if edge < min_edge: reasons.append(f"edge {edge:.1f}<{min_edge:.1f} pp")
        if ev < min_ev: reasons.append(f"EV {ev:.1f}<{min_ev:.1f}%")
        rows.append({
            "Seleccion": name,
            "Prob_Modelo": round(float(prob), 1),
            "Prob_Condicional": round(cond, 1),
            "Push": round(float(push), 1),
            "Cuota": round(float(odd), 3),
            "Prob_Mercado_NoVig": round(float(market_p), 1),
            "Edge_pp": round(edge, 1),
            "EV_pct": round(ev, 1),
            "Estado": "VALUE BET" if not reasons else "NO BET",
            "Motivo": "Supera filtros" if not reasons else "; ".join(reasons),
        })
    return rows


def analizar_apuestas_mlb(res_mc, preds_ml, cuotas_reales, linea_carreras):
    """Compatibilidad V2. Combina MC+ML y evalúa contra mercado sin vig."""
    rows = []
    ml_h = float(preds_ml.get("Probabilidad_Local", 50.0))
    ml_a = float(preds_ml.get("Probabilidad_Visita", 50.0))
    mc_h = float(res_mc["Moneyline"]["Gana Local"])
    mc_a = float(res_mc["Moneyline"]["Gana Visita"])
    p_h = 0.55 * ml_h + 0.45 * mc_h
    p_a = 0.55 * ml_a + 0.45 * mc_a

    oh, oa = cuotas_reales.get("Moneyline_Local"), cuotas_reales.get("Moneyline_Visita")
    rows += evaluar_dos_vias("Moneyline Local", "Moneyline Visitante", p_h, p_a, oh, oa,
                             min_edge=3.0, min_ev=2.0, min_prob=52.0)

    line = float(linea_carreras)
    mc_over = float(res_mc["Carreras"].get(f"Over {line}", 50.0))
    mc_under = float(res_mc["Carreras"].get(f"Under {line}", 50.0))
    push = float(res_mc["Carreras"].get(f"Push {line}", 0.0))
    ml_over = preds_ml.get("Prob_Over")
    ml_under = preds_ml.get("Prob_Under")
    if ml_over is not None and ml_under is not None:
        p_over = 0.55 * float(ml_over) + 0.45 * mc_over
        p_under = 0.55 * float(ml_under) + 0.45 * mc_under
        rows += evaluar_dos_vias(
            f"Over {line}", f"Under {line}", p_over, p_under,
            cuotas_reales.get("Cuota_Over"), cuotas_reales.get("Cuota_Under"),
            push_a=push, push_b=push, min_edge=4.0, min_ev=3.0, min_prob=57.0,
        )
    return pd.DataFrame(rows)
