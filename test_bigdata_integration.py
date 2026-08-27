import tempfile
from pathlib import Path

import pandas as pd

from modules.bigdata_mlb import MLBDataWarehouse, LEGACY_ML_COLUMNS, bootstrap_from_repository
from modules.bigdata_tracking import sync_snapshot_rows, settle_snapshot_rows
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


def test_warehouse_auto_refresh_detects_historical_correction():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / 'warehouse'
        source = Path(td) / 'games.csv'
        games = pd.read_csv('data/mlb_games.csv').head(1200).copy()
        games.to_csv(source, index=False)
        wh = MLBDataWarehouse(root)
        expected_initial = len(wh._normalize_games(games))
        assert expected_initial > 100
        first = wh.ensure_fresh_from_repository(source)
        assert first['fresh'] and first['rebuilt']
        assert first['features'] == expected_initial
        second = wh.ensure_fresh_from_repository(source)
        assert second['fresh'] and not second['rebuilt']

        # Same row count/date range: changing valid scores must invalidate the store.
        score_col = 'Home_Score' if 'Home_Score' in games.columns else 'home_score'
        scores = pd.to_numeric(games[score_col], errors='coerce')
        valid_scores = scores.notna()
        assert valid_scores.any()
        games.loc[valid_scores, score_col] = scores.loc[valid_scores] + 1.0
        games.to_csv(source, index=False)
        corrected = wh.ensure_fresh_from_repository(source)
        assert corrected['fresh'] and corrected['rebuilt']

        # Removing a source row that survives normalization must also remove it from DuckDB/Parquet.
        normalized_before_remove = wh._normalize_games(games)
        assert not normalized_before_remove.empty
        target_key = normalized_before_remove.iloc[len(normalized_before_remove) // 2]['game_key']
        # Find a raw row whose normalized key is the selected target and remove that exact source game.
        raw_with_keys = wh._normalize_games(games)
        target = raw_with_keys[raw_with_keys['game_key'] == target_key].iloc[0]
        game_pk_col = 'gamePk' if 'gamePk' in games.columns else None
        if game_pk_col and str(target_key).startswith('pk:'):
            target_pk = int(str(target_key).split(':', 1)[1])
            raw_pk = pd.to_numeric(games[game_pk_col], errors='coerce')
            drop_idx = games.index[raw_pk == target_pk]
        else:
            date_col = 'Date' if 'Date' in games.columns else 'date'
            home_col = 'Home' if 'Home' in games.columns else 'home'
            away_col = 'Away' if 'Away' in games.columns else 'away'
            raw_date = pd.to_datetime(games[date_col], errors='coerce').dt.strftime('%Y-%m-%d')
            drop_idx = games.index[(raw_date == pd.Timestamp(target['Date']).strftime('%Y-%m-%d')) &
                                   (games[home_col].map(lambda x: str(x)) == str(target['Home'])) &
                                   (games[away_col].map(lambda x: str(x)) == str(target['Away']))]
        assert len(drop_idx) >= 1
        games = games.drop(drop_idx[0]).reset_index(drop=True)
        games.to_csv(source, index=False)
        removed = wh.ensure_fresh_from_repository(source)
        assert removed['fresh'] and removed['rebuilt']

        expected = len(wh._normalize_games(games))
        con = wh.connect()
        try:
            rebuilt_games = con.execute('SELECT COUNT(*) FROM games').fetchone()[0]
            rebuilt_features = con.execute('SELECT COUNT(*) FROM pregame_features').fetchone()[0]
        finally:
            con.close()
        assert rebuilt_games == expected
        assert rebuilt_features == expected


def test_scanner_bridge_is_idempotent_and_settles():
    row = {
        'snapshot_utc': '2026-08-27T12:00:00+00:00', 'game_date': '2099-01-01',
        'game_pk': 999999999, 'away': 'BOS', 'home': 'NYY', 'market': 'Moneyline',
        'selection': 'Gana Local (New York Yankees)', 'line': None, 'odds': 1.91,
        'prob_ml': 61.0, 'prob_mc': 60.0, 'prob_combined': 60.5,
        'edge_pp': 6.0, 'ev_pct': 15.55, 'kelly_pct': 4.0, 'model_version': 'v6-test',
        'result_status': 'pending', 'profit_units': None,
    }
    a = sync_snapshot_rows([row]); b = sync_snapshot_rows([row])
    assert a['ok'] and b['ok']
    wh = MLBDataWarehouse(); con = wh.connect()
    try:
        count = con.execute("SELECT COUNT(*) FROM predictions WHERE game_key='pk:999999999'").fetchone()[0]
        assert count == 1
    finally:
        con.close()
    row['result_status'] = 'win'; row['profit_units'] = 0.91
    s = settle_snapshot_rows([row])
    assert s['ok'] and s['settled'] == 1
    con = wh.connect()
    try:
        result = con.execute("SELECT result, profit_units FROM predictions WHERE game_key='pk:999999999'").fetchone()
        assert result[0] == 'WIN' and abs(float(result[1]) - 0.91) < 1e-9
    finally:
        con.close()


if __name__ == '__main__':
    test_warehouse_training_contract_and_tracking()
    test_default_repository_bootstrap_and_predictor_uses_store()
    test_warehouse_auto_refresh_detects_historical_correction()
    test_scanner_bridge_is_idempotent_and_settles()
    print('Big Data integration: OK')
