import pandas as pd


def _decimal(value):
    try:
        v = float(value)
        return v if v > 1.0 else None
    except (TypeError, ValueError):
        return None


def _no_vig_two_way(odds_a, odds_b):
    a, b = _decimal(odds_a), _decimal(odds_b)
    if a is None or b is None:
        return None, None
    ia, ib = 1.0 / a, 1.0 / b
    total = ia + ib
    if total <= 0:
        return None, None
    return ia / total, ib / total


def _ev(prob_win, odds, prob_push=0.0):
    o = _decimal(odds)
    if o is None:
        return None
    p = max(0.0, min(1.0, float(prob_win)))
    push = max(0.0, min(1.0 - p, float(prob_push)))
    lose = max(0.0, 1.0 - p - push)
    return p * (o - 1.0) - lose


def _conditional_no_push(prob_win, prob_push=0.0):
    """Probability of winning conditional on the bet being decided (not a push)."""
    p = max(0.0, min(1.0, float(prob_win)))
    push = max(0.0, min(1.0 - p, float(prob_push)))
    decided = 1.0 - push
    if decided <= 1e-12:
        return None
    return p / decided


def _blend(a, b, wa=0.55):
    if a is None:
        return b
    if b is None:
        return a
    return wa * float(a) + (1.0 - wa) * float(b)


def analizar_apuestas_mlb(res_mc, preds_ml, cuotas_reales, linea_carreras):
    """Analyze moneyline and totals using consistent probability definitions.

    EV uses the unconditional win/lose/push distribution. Two-way no-vig market
    probabilities are conditional on the wager being decided, so edge comparisons
    for integer totals also use the model probability conditional on no push.
    """
    apuestas = []

    mc_loc = float(res_mc["Moneyline"]["Gana Local"]) / 100.0
    mc_vis = float(res_mc["Moneyline"]["Gana Visita"]) / 100.0
    ml_loc = None
    ml_vis = None
    if isinstance(preds_ml, dict):
        if preds_ml.get("Probabilidad_Local") is not None:
            ml_loc = float(preds_ml["Probabilidad_Local"]) / 100.0
        if preds_ml.get("Probabilidad_Visita") is not None:
            ml_vis = float(preds_ml["Probabilidad_Visita"]) / 100.0

    prob_loc = _blend(ml_loc, mc_loc)
    prob_vis = _blend(ml_vis, mc_vis)
    s = prob_loc + prob_vis
    if s > 0:
        prob_loc, prob_vis = prob_loc / s, prob_vis / s

    key_over = f"Over {linea_carreras}"
    key_under = f"Under {linea_carreras}"
    key_push = f"Push {linea_carreras}"
    prob_over = float(res_mc["Carreras"].get(key_over, 0.0)) / 100.0
    prob_under = float(res_mc["Carreras"].get(key_under, 0.0)) / 100.0
    prob_push = float(res_mc["Carreras"].get(key_push, 0.0)) / 100.0

    cuota_loc = _decimal(cuotas_reales.get("Moneyline_Local"))
    cuota_vis = _decimal(cuotas_reales.get("Moneyline_Visita"))
    cuota_over = _decimal(cuotas_reales.get("Cuota_Over"))
    cuota_under = _decimal(cuotas_reales.get("Cuota_Under"))

    mkt_loc, mkt_vis = _no_vig_two_way(cuota_loc, cuota_vis)
    mkt_over, mkt_under = _no_vig_two_way(cuota_over, cuota_under)

    def add_row(name, p, odds, market_p=None, push=0.0):
        if odds is None:
            return
        ev = _ev(p, odds, push)
        model_edge_p = _conditional_no_push(p, push) if push > 0 else p
        edge = None if market_p is None or model_edge_p is None else model_edge_p - market_p
        strong = ev is not None and ev >= 0.03 and (edge is None or edge >= 0.025)
        fair = ev is not None and ev > 0 and (edge is None or edge > 0)
        apuestas.append({
            "Seleccion": name,
            "Prob Model": f"{round(p * 100, 1)}%",
            "Prob Model Sin Push": None if push <= 0 or model_edge_p is None else f"{round(model_edge_p * 100, 1)}%",
            "Prob Push": None if push <= 0 else f"{round(push * 100, 1)}%",
            "Cuota": odds,
            "Prob Mercado No-Vig": None if market_p is None else f"{round(market_p * 100, 1)}%",
            "Edge": None if edge is None else f"{round(edge * 100, 2)} pp",
            "EV+": "N/D" if ev is None else f"{round(ev * 100, 2)}%",
            "Veredicto": "🔥 ¡Apostar!" if strong else ("✅ Valor Justo" if fair else "❌ Sin Valor"),
        })

    add_row("Moneyline Local", prob_loc, cuota_loc, mkt_loc)
    add_row("Moneyline Visitante", prob_vis, cuota_vis, mkt_vis)
    add_row(f"Over {linea_carreras} Carreras", prob_over, cuota_over, mkt_over, prob_push)
    add_row(f"Under {linea_carreras} Carreras", prob_under, cuota_under, mkt_under, prob_push)

    return pd.DataFrame(apuestas)
