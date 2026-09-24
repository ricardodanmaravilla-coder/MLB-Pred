import os
from pathlib import Path

from modules.web_service import american_to_decimal, estimate_ml_probability, kelly_fraction_pct
from modules.game_context import market_from_event
from modules.scanner_engine import total_candidate
from web_app import app


def test_cloud_run_routes_exist():
    paths={route.path for route in app.routes}
    for required in ('/','/api/health','/api/slate','/api/scan','/api/reload'):
        assert required in paths


def test_static_frontend_exists_and_calls_api():
    html=Path('static/index.html').read_text(encoding='utf-8')
    assert '/api/health' in html
    assert '/api/slate' in html
    assert '/api/scan' in html
    assert 'MLB Quant Analytics V7' in html


def test_cloudrun_port_is_runtime_driven():
    docker=Path('Dockerfile').read_text(encoding='utf-8')
    assert '${PORT}' in docker
    assert 'uvicorn web_app:app' in docker


def test_market_math_matches_expected_contract():
    assert american_to_decimal(-110)==1.91
    assert american_to_decimal(120)==2.2
    assert estimate_ml_probability(9.0,8.5,'over',3.5)>50.0
    assert estimate_ml_probability(8.0,8.5,'under',3.5)>50.0
    assert 0.0<=kelly_fraction_pct(60,1.91)<=25.0


def test_no_embedded_odds_secret():
    source=Path('modules/web_service.py').read_text(encoding='utf-8')
    assert 'ODDS_API_KEY = os.getenv' in source
    assert 'apiKey=' in source
    assert 'ODDS_API_KEY = "' not in source


if __name__=='__main__':
    tests=[v for k,v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for test in tests:
        test(); print('PASS',test.__name__)
    print('Cloud Run web tests passed:',len(tests))


def test_implausible_market_odds_are_rejected():
    event={
        'home_team':'Baltimore Orioles','away_team':'Toronto Blue Jays',
        'bookmakers':[{'title':'bad-feed','markets':[
            {'key':'h2h','outcomes':[{'name':'Baltimore Orioles','price':-110},{'name':'Toronto Blue Jays','price':-110}]},
            {'key':'totals','outcomes':[{'name':'Over','price':5000,'point':7.5},{'name':'Under','price':-110,'point':7.5}]},
        ]}]
    }
    market=market_from_event(event, american_to_decimal)
    assert market['cuota_over'] is None and market['linea_carreras'] is None
    assert total_candidate('Over 7.5',65,65,51.0,.5) is None



def test_totals_survive_when_book_has_no_moneyline():
    event={'home_team':'Baltimore Orioles','away_team':'Toronto Blue Jays','bookmakers':[{'title':'totals-only','markets':[{'key':'totals','outcomes':[{'name':'Over','price':-110,'point':7.5},{'name':'Under','price':-110,'point':7.5}]}]}]}
    market=market_from_event(event, american_to_decimal)
    assert market['linea_carreras'] == 7.5
    assert market['cuota_over'] is not None and market['cuota_under'] is not None

