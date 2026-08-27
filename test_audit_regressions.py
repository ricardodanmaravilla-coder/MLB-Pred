import pandas as pd

from modules.bigdata_mlb import MLBDataWarehouse
from modules.historical_mlb import prepare_games
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


if __name__ == '__main__':
    test_gameid_survives_as_gamepk_and_doubleheaders_do_not_collapse()
    test_total_edge_is_no_push_conditional_while_ev_is_unconditional()
    print('Audit regressions: OK')
