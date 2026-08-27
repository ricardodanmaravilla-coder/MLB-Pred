import os
from pathlib import Path

from modules.web_service import american_to_decimal, estimate_ml_probability, kelly_fraction_pct
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
