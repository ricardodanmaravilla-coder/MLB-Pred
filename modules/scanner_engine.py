"""Pure market-selection rules shared by Streamlit and backtests.

V6 supports a forward-calibrated ML/Monte Carlo blend while preserving a conservative
50/50 default. Pushes on integer lines are treated as sportsbook refunds.
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
    blend_weight_ml: float
    odds: float
    market_no_vig: Optional[float]
    edge_pp: Optional[float]
    ev_pct: float
    disagreement_pp: float
    score: float
    accepted: bool
    reason: str
    def to_dict(self): return asdict(self)


def no_vig_two_way(odds_a, odds_b):
    try:
        a,b=float(odds_a),float(odds_b)
        if a<=1.0 or b<=1.0: return None,None
        ia,ib=1/a,1/b; total=ia+ib; return ia/total,ib/total
    except (TypeError,ValueError,ZeroDivisionError): return None,None


def _candidate(market,selection,prob_ml,prob_mc,odds,market_no_vig,min_ml,min_mc,min_combined,max_disagreement,min_edge_pp,min_ev_pct,push_pct=0.0,blend_weight_ml=0.50):
    try:
        pml=float(prob_ml); pmc=float(prob_mc); o=float(odds); push=max(0.,min(99.,float(push_pct))); w=float(blend_weight_ml)
    except (TypeError,ValueError): return None
    if o<=1.: return None
    # Never allow a transient calibration artifact to silence either engine.
    w=max(.30,min(.70,w)); combined=w*pml+(1-w)*pmc; disagreement=abs(pml-pmc)
    # Only Monte Carlo contains discrete push mass, so its contribution follows MC weight.
    model_push=push*(1-w); p_win=combined/100.; p_push=model_push/100.
    if p_win+p_push>1.: p_win=max(0.,1.-p_push); combined=p_win*100.
    p_loss=max(0.,1.-p_win-p_push); ev_pct=(p_win*(o-1.)-p_loss)*100.
    decision_mass=max(1e-9,1.-p_push); model_decision_prob=p_win/decision_mass
    edge_pp=None if market_no_vig is None else (model_decision_prob-float(market_no_vig))*100.
    checks=[(pml>=min_ml,f"ML {pml:.1f}% < {min_ml:.1f}%"),(pmc>=min_mc,f"MC {pmc:.1f}% < {min_mc:.1f}%"),
            (combined>=min_combined,f"Combinada {combined:.1f}% < {min_combined:.1f}%"),(disagreement<=max_disagreement,f"Desacuerdo {disagreement:.1f} pp > {max_disagreement:.1f}"),
            (ev_pct>=min_ev_pct,f"EV {ev_pct:.1f}% < {min_ev_pct:.1f}%")]
    if market_no_vig is not None: checks.append((edge_pp>=min_edge_pp,f"Edge {edge_pp:.1f} pp < {min_edge_pp:.1f} pp"))
    fails=[msg for ok,msg in checks if not ok]; accepted=not fails; edge_component=0. if edge_pp is None else edge_pp; score=1.5*edge_component+ev_pct-.15*disagreement
    return Candidate(market,selection,round(pml,3),round(pmc,3),round(combined,3),round(model_push,3),round(w,3),round(o,4),
                     None if market_no_vig is None else round(float(market_no_vig)*100.,3),None if edge_pp is None else round(edge_pp,3),round(ev_pct,3),
                     round(disagreement,3),round(score,4),accepted,'Cumple filtros' if accepted else '; '.join(fails))


def moneyline_candidate(selection,prob_ml,prob_mc,odds,market_no_vig=None,blend_weight_ml=.50):
    return _candidate('Moneyline',selection,prob_ml,prob_mc,odds,market_no_vig,55.,55.,55.,15.,2.5,3.,blend_weight_ml=blend_weight_ml)

def total_candidate(selection,prob_ml,prob_mc,odds,market_no_vig=None,prob_push_mc=0.0,blend_weight_ml=.50):
    is_over=str(selection).strip().lower().startswith('over'); min_ml=54. if is_over else 52.
    return _candidate('Totales',selection,prob_ml,prob_mc,odds,market_no_vig,min_ml,52.,54.,15.,4.,4.,push_pct=prob_push_mc,blend_weight_ml=blend_weight_ml)

def runline_candidate(selection,prob_ml,prob_mc,odds,market_no_vig=None,prob_push_mc=0.0,blend_weight_ml=.50):
    return _candidate('Hándicap',selection,prob_ml,prob_mc,odds,market_no_vig,58.,58.,60.,10.,6.,6.,push_pct=prob_push_mc,blend_weight_ml=blend_weight_ml)

def top_candidates(candidates,limit=3):
    valid=[c for c in candidates if c is not None and c.accepted]; valid.sort(key=lambda c:c.score,reverse=True); return valid[:int(limit)]
