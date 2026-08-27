import tempfile
from pathlib import Path

import pandas as pd

from modules.bigdata_mlb import MLBDataWarehouse, LEGACY_ML_COLUMNS, bootstrap_from_repository
from modules.ml_mlb import PredictorMLMLB


def test_warehouse_training_contract_and_tracking():
    with tempfile.TemporaryDirectory() as td:
        wh = MLBDataWarehouse(Path(td) / 'warehouse')
        games = pd.read_csv('data/mlb_games.csv')
        bat = pd.read_csv('data/mlb_batting.csv')
        pit = pd.read_csv('data/mlb_pitching.csv')
        assert wh.ingest_games(games) > 1000
        assert wh.rebuild_feature_store() > 1000
        frame = wh.legacy_ml_training_frame(bat, pit)
        assert len(frame) > 1000
        assert all(c in frame.columns for c in LEGACY_ML_COLUMNS)
        assert frame[LEGACY_ML_COLUMNS].notna().all().all()
        pid = wh.record_prediction(
            game_key='test:BOS@NYY', game_date='2026-08-27', home='NYY', away='BOS',
            market='Moneyline', selection='NYY', prob_ml=61, prob_mc=60,
            probability=60.5, odds=1.90, edge_pp=6.0, ev_pct=14.95,
            kelly_pct=4.0, accepted=True,
        )
        assert wh.settle_prediction(pid, 'WIN', 0.90)
        s = wh.performance_summary()
        assert s['settled'] == 1 and s['wins'] == 1 and s['profit_units'] == 0.9


def test_default_repository_bootstrap_and_predictor_uses_store():
    status = bootstrap_from_repository()
    assert status['features'] > 1000
    bat = pd.read_csv('data/mlb_batting.csv')
    pit = pd.read_csv('data/mlb_pitching.csv')
    games = pd.read_csv('data/mlb_games.csv')
    m = PredictorMLMLB()
    assert m.entrenar(bat, pit, games)
    assert m.training_source == 'duckdb_parquet_feature_store'
    r = m.predecir_partido('NYY', 'BOS', 100, 100, 4.1, 4.2)
    assert r['Training_Source'] == 'duckdb_parquet_feature_store'
    assert 0 <= r['Probabilidad_Local'] <= 100


if __name__ == '__main__':
    test_warehouse_training_contract_and_tracking()
    test_default_repository_bootstrap_and_predictor_uses_store()
    print('Big Data integration: OK')
