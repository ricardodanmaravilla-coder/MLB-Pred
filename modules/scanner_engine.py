"""Pure market-selection rules shared by Streamlit and backtests.

Production and validation use one source of truth. Totals/run lines correctly
account for sportsbook pushes on integer lines instead of treating refunds as losses.
"""

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Candidate:
    market: str
    selection: str
    prob_ml: float
    prob_mc: float
    probability: float
    push_probability: float
    odds: float
    market_no_vig: Optional[float]
    edge_pp: Optional[float]
    ev_pct: float
    disagreement_pp: float
    score: float
    accepted: bool
    reason: str

    def to_dict(self):
        return asdict(self)


def no_vig_two_way(odds_a, odds_b):
    try:
        a, b = float(odds_a), float(odds_b)
        if a <= 1.0 or b <= 1.0:
            return None, None
        ia, ib = 1.0 / a, 1.0 / b
        total = ia + ib
        return ia / total, ib / total
    except (TypeError, ValueError, ZeroDivisionError):
        return None, None


def _candidate(market, selection, prob_ml, prob_mc, odds, market_no_vig,
               min_ml, min_mc, min_combined, max_disagreement,
               min_edge_pp, min_ev_pct, push_pct=0.0):
    try:
        pml = float(prob_ml)
        pmc = float(prob_mc)
        o = float(odds)
        push = max(0.0, min(99.0, float(push_pct)))
    except (TypeError, ValueError):
        return None
    if o <= 1.0:
        return None

    combined = (pml + pmc) / 2.0
    disagreement = abs(pml - pmc)
    # ML uses a continuous distribution (zero push mass), while MC is discrete.
    # When push_pct comes from MC, half of it belongs to the 50/50 blended model.
    model_push = push / 2.0
    p_win = combined / 100.0
    p_push = model_push / 100.0
    if p_win + p_push > 1.0:
        p_win = max(0.0, 1.0 - p_push)
        combined = p_win * 100.0
    p_loss = max(0.0, 1.0 - p_win - p_push)
    ev_pct = (p_win * (o - 1.0) - p_loss) * 100.0

    # Sportsbook two-way no-vig probabilities are conditional on a graded decision.
    decision_mass = max(1e-9, 1.0 - p_push)
    model_decision_prob = p_win / decision_mass
    edge_pp = None if market_no_vig is None else (model_decision_prob - float(market_no_vig)) * 100.0

    checks = [
        (pml >= min_ml, f"ML {pml:.1f}% < {min_ml:.1f}%"),
        (pmc >= min_mc, f"MC {pmc:.1f}% < {min_mc:.1f}%"),
        (combined >= min_combined, f"Combinada {combined:.1f}% < {min_combined:.1f}%"),
        (disagreement <= max_disagreement, f"Desacuerdo {disagreement:.1f} pp > {max_disagreement:.1f}"),
        (ev_pct >= min_ev_pct, f"EV {ev_pct:.1f}% < {min_ev_pct:.1f}%"),
    ]
    if market_no_vig is not None:
        checks.append((edge_pp >= min_edge_pp, f"Edge {edge_pp:.1f} pp < {min_edge_pp:.1f} pp"))

    fails = [msg for ok, msg in checks if not ok]
    accepted = not fails
    edge_component = 0.0 if edge_pp is None else edge_pp
    score = (1.5 * edge_component) + ev_pct - (0.15 * disagreement)

    return Candidate(
        market=market, selection=selection, prob_ml=round(pml,3), prob_mc=round(pmc,3),
        probability=round(combined,3), push_probability=round(model_push,3), odds=round(o,4),
        market_no_vig=None if market_no_vig is None else round(float(market_no_vig)*100.0,3),
        edge_pp=None if edge_pp is None else round(edge_pp,3), ev_pct=round(ev_pct,3),
        disagreement_pp=round(disagreement,3), score=round(score,4), accepted=accepted,
        reason='Cumple filtros' if accepted else '; '.join(fails),
    )


def moneyline_candidate(selection, prob_ml, prob_mc, odds, market_no_vig=None):
    return _candidate('Moneyline', selection, prob_ml, prob_mc, odds, market_no_vig,
                      min_ml=55.0, min_mc=55.0, min_combined=55.0,
                      max_disagreement=15.0, min_edge_pp=2.5, min_ev_pct=3.0)


def total_candidate(selection, prob_ml, prob_mc, odds, market_no_vig=None, prob_push_mc=0.0):
    is_over = str(selection).strip().lower().startswith('over')
    min_ml = 54.0 if is_over else 52.0
    return _candidate('Totales', selection, prob_ml, prob_mc, odds, market_no_vig,
                      min_ml=min_ml, min_mc=52.0, min_combined=54.0,
                      max_disagreement=15.0, min_edge_pp=4.0, min_ev_pct=4.0,
                      push_pct=prob_push_mc)


def runline_candidate(selection, prob_ml, prob_mc, odds, market_no_vig=None, prob_push_mc=0.0):
    return _candidate('Hándicap', selection, prob_ml, prob_mc, odds, market_no_vig,
                      min_ml=58.0, min_mc=58.0, min_combined=60.0,
                      max_disagreement=10.0, min_edge_pp=6.0, min_ev_pct=6.0,
                      push_pct=prob_push_mc)


def top_candidates(candidates, limit=3):
    valid = [c for c in candidates if c is not None and c.accepted]
    valid.sort(key=lambda c: c.score, reverse=True)
    return valid[: int(limit)]
