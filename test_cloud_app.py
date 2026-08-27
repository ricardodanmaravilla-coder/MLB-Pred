from pathlib import Path

def test_cloud_files_exist():
    for p in ('cloud_app.py','modules/cloud_service.py','web/index.html','Dockerfile','cloudbuild.yaml'):
        assert Path(p).exists(), p

def test_no_streamlit_in_cloud_entrypoint():
    text=Path('cloud_app.py').read_text(encoding='utf-8')
    assert 'streamlit' not in text.lower()
    assert 'FastAPI' in text

def test_cloud_run_port_contract():
    docker=Path('Dockerfile').read_text(encoding='utf-8')
    assert '${PORT}' in docker and '0.0.0.0' in docker

def test_service_has_required_routes():
    text=Path('cloud_app.py').read_text(encoding='utf-8')
    for route in ('/api/health','/api/games','/api/analyze/{game_pk}','/api/scan'):
        assert route in text
