from pathlib import Path

import pandas as pd

from modules.blend_calibration import learn_blend_weight
from modules.ml_mlb import preferred_batting_column, preferred_pitching_column
from modules.montecarlo_mlb import _starter_share, simular_partido_mlb
from modules.scanner_engine import moneyline_candidate


def test_real_metric_preference():
    teams=[f'T{i}' for i in range(30)]
    bat=pd.DataFrame({'Team':teams,'Season':[2025]*30,'OPS_Index':[75]*30,'wRC+':[100+i%5 for i in range(30)],'wRC+_Source':['FANGRAPHS_REAL_WRCPLUS']*30})
    pit=pd.DataFrame({'Team':teams,'Season':[2025]*30,'ERA':[4.2]*30,'xFIP':[3.8+i*.01 for i in range(30)],'xFIP_Source':['FANGRAPHS_REAL_XFIP']*30})
    assert preferred_batting_column(bat)=='wRC+'
    assert preferred_pitching_column(pit)=='xFIP'
    bat['wRC+_Source']='LEGACY_OPS_X100_NOT_REAL_WRCPLUS'; pit['xFIP_Source']='LEGACY_ERA_PROXY_NOT_REAL_XFIP'
    assert preferred_batting_column(bat)=='OPS_Index'
    assert preferred_pitching_column(pit)=='ERA'


def test_blend_weight_guardrails():
    small=pd.DataFrame({'market':['Moneyline']*20,'result_status':['win','loss']*10,'prob_ml':['60%']*20,'prob_mc':['55%']*20})
    w,n,src=learn_blend_weight('Moneyline',small); assert w==.5 and n==20 and src=='default_small_sample'
    # Construct 100 decisions where ML tracks outcomes much better than MC.
    rows=[]
    for i in range(100):
        win=i%2==0
        rows.append({'market':'Moneyline','result_status':'win' if win else 'loss','prob_ml':'75%' if win else '25%','prob_mc':'52%' if win else '48%'})
    w,n,src=learn_blend_weight('Moneyline',pd.DataFrame(rows)); assert n==100 and .50<w<=.70 and src=='forward_brier_shrunk'


def test_candidate_uses_requested_blend():
    c=moneyline_candidate('Home',70,50,2.0,.50,blend_weight_ml=.65)
    assert c is not None and abs(c.probability-63.0)<1e-9 and abs(c.blend_weight_ml-.65)<1e-9


def test_starter_workload_changes_run_environment():
    assert abs(_starter_share(None)-(5.2/9.0))<1e-9
    assert abs(_starter_share(6.3)-.70)<1e-9
    base=dict(local='NYY',visita='BOS',pitcher_loc_xfip=4.0,pitcher_vis_xfip=6.0,wrc_loc=100,wrc_vis=100,
              bullpen_loc_era=4.0,bullpen_vis_era=3.0,park_factor=100,altitud_ft=50,viento_mph=0,direccion_viento='None',temp_f=72,
              linea_carreras_casino=8.5,df_games=pd.DataFrame(),num_simulaciones=6000,starter_ip_loc=5.2)
    short=simular_partido_mlb(**base,starter_ip_vis=4.0)
    long=simular_partido_mlb(**base,starter_ip_vis=6.3)
    assert long['Metadatos']['Carreras_Exp_Local']>short['Metadatos']['Carreras_Exp_Local']
    assert long['Metadatos']['StarterShare_Visita']>short['Metadatos']['StarterShare_Visita']
    assert long['Metadatos']['Pitching_Agregado_Es_Proxy_Bullpen'] is True


def test_app_v6_integration_present():
    text=Path('app_mlb.py').read_text(encoding='utf-8')
    for marker in ('MLB Quant Analytics Pro V6','preferred_batting_column','preferred_pitching_column','market_blend_weights','starter_ip_loc=starter_ip_loc','blend_weight_ml='):
        assert marker in text, marker
    assert "'model_version': 'v6'" in text


def test_miners_expose_real_sources_and_fallbacks():
    a=Path('minero_mlb.py').read_text(encoding='utf-8'); b=Path('minero_pitchers.py').read_text(encoding='utf-8')
    assert 'FANGRAPHS_REAL_WRCPLUS' in a and 'FANGRAPHS_REAL_XFIP' in a and 'MLB_STATSAPI_FALLBACK' in a
    assert 'FANGRAPHS_REAL_XFIP' in b and 'FANGRAPHS_REAL_RELIEVER_XFIP' in b


def main():
    tests=[v for k,v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for fn in tests:
        fn(); print('PASS',fn.__name__)
    print(f'V6 predictive tests passed: {len(tests)}')

if __name__=='__main__': main()
