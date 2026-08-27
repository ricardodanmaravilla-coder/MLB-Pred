import numpy as np
import pandas as pd

from modules.bigdata_mlb import MLBDataWarehouse
from modules.historical_mlb import prepare_games
from modules.ml_mlb import PredictorMLMLB, _date_safe_cut
from modules.odds_mlb import _conditional_no_push, _ev, analizar_apuestas_mlb


def test_gameid_survives_as_gamepk_and_doubleheaders_do_not_collapse():
    raw = pd.DataFrame([
        {'GameID': 1001, 'Date': '2026-06-10', 'Season': 2026, 'GameType': 'R', 'Away': 'BOS', 'Home': 'NYY', 'Away_Score': 2, 'Home_Score': 3},
        {'GameID': 1002, 'Date': '2026-06-10', 'Season': 2026, 'GameType': 'R', 'Away': 'BOS', 'Home': 'NYY', 'Away_Score': 5, 'Home_Score': 1},
    ])
    prepared = prepare_games(raw)
    assert len(prepared) == 2
    assert 'gamePk' in prepared.columns
    assert prepared['gamePk'].tolist() == [1001, 1002]
    normalized = MLBDataWarehouse._normalize_games(raw)
    assert len(normalized) == 2
    assert normalized['game_key'].tolist() == ['pk:1001', 'pk:1002']


def test_total_edge_is_no_push_conditional_while_ev_is_unconditional():
    assert abs(_conditional_no_push(0.48, 0.08) - 0.48 / 0.92) < 1e-12
    assert abs(_ev(0.48, 2.0, 0.08) - 0.04) < 1e-12
    res_mc = {
        'Moneyline': {'Gana Local': 50.0, 'Gana Visita': 50.0},
        'Carreras': {'Over 8.0': 48.0, 'Under 8.0': 44.0, 'Push 8.0': 8.0},
    }
    out = analizar_apuestas_mlb(
        res_mc,
        {'Probabilidad_Local': 50.0, 'Probabilidad_Visita': 50.0},
        {'Moneyline_Local': 1.91, 'Moneyline_Visita': 1.91, 'Cuota_Over': 2.0, 'Cuota_Under': 2.0},
        8.0,
    )
    over = out[out['Seleccion'].str.startswith('Over')].iloc[0]
    assert over['Prob Model'] == '48.0%'
    assert over['Prob Model Sin Push'] == '52.2%'
    assert over['Prob Push'] == '8.0%'
    assert over['Edge'] == '2.17 pp'
    assert over['EV+'] == '4.0%'


def test_csv_fallback_doubleheader_features_do_not_see_game_one_result():
    games = prepare_games(pd.DataFrame([
        {'GameID': 2001, 'Date': '2026-06-10', 'Season': 2026, 'GameType': 'R', 'Away': 'BOS', 'Home': 'NYY', 'Away_Score': 1, 'Home_Score': 10},
        {'GameID': 2002, 'Date': '2026-06-10', 'Season': 2026, 'GameType': 'R', 'Away': 'BOS', 'Home': 'NYY', 'Away_Score': 9, 'Home_Score': 0},
    ]))
    bat = pd.DataFrame([
        {'Team':'NYY','Season':2025,'OPS_Index':100.0}, {'Team':'BOS','Season':2025,'OPS_Index':100.0},
    ])
    pit = pd.DataFrame([
        {'Team':'NYY','Season':2025,'ERA':4.1}, {'Team':'BOS','Season':2025,'ERA':4.1},
    ])
    m = PredictorMLMLB()
    bd = {('NYY',2025):100.0, ('BOS',2025):100.0}
    pdict = {('NYY',2025):4.1, ('BOS',2025):4.1}
    X, yw, yr, yd, dates = m._training_arrays(bat, pit, games, bd, pdict)
    assert X.shape == (2, 20)
    np.testing.assert_allclose(X[0, :15], X[1, :15])
    assert m.training_source == 'csv_chronological_daily_safe'
    assert len(set(pd.to_datetime(dates).date)) == 1


def test_internal_validation_cut_never_splits_calendar_date():
    dates = pd.to_datetime(
        ['2026-04-01'] * 400 + ['2026-04-02'] * 400 + ['2026-04-03'] * 250 + ['2026-04-04'] * 250
    )
    cut = _date_safe_cut(dates, ratio=0.80, min_train=100, min_validation=50)
    train_days = set(pd.to_datetime(dates[:cut]).date)
    validation_days = set(pd.to_datetime(dates[cut:]).date)
    assert train_days
    assert validation_days
    assert train_days.isdisjoint(validation_days)


if __name__ == '__main__':
    test_gameid_survives_as_gamepk_and_doubleheaders_do_not_collapse()
    test_total_edge_is_no_push_conditional_while_ev_is_unconditional()
    test_csv_fallback_doubleheader_features_do_not_see_game_one_result()
    test_internal_validation_cut_never_splits_calendar_date()
    print('Audit regressions: OK')
