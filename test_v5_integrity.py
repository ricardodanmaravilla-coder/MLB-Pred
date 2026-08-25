from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from modules.game_context import (
    slate_date, park_for_team, match_odds_game, conservative_auto_weather, market_from_event,
)
from modules.historical_mlb import prepare_games
from modules.ml_mlb import _frame_signature
from modules.montecarlo_mlb import simular_partido_mlb
from modules.scanner_engine import total_candidate
from modules.team_utils import normalize_team
from settle_picks import settle_row


EXPECTED_TEAMS = {
    'NYY','BOS','LAD','HOU','ATL','PHI','BAL','TB','TOR','CWS','CLE','DET','KC','MIN','LAA',
    'OAK','SEA','TEX','CHC','CIN','MIL','PIT','STL','AZ','COL','SF','SD','MIA','NYM','WSH'
}


def test_team_aliases_and_parks():
    assert normalize_team('ARI') == 'AZ'
    assert normalize_team('SDP') == 'SD'
    assert normalize_team('SFG') == 'SF'
    assert normalize_team('CHW') == 'CWS'
    assert normalize_team('WSN') == 'WSH'
    parks = pd.read_csv('data/mlb_park_factors.csv')
    resolved = {team for team in EXPECTED_TEAMS if park_for_team(parks, team) is not None}
    assert resolved == EXPECTED_TEAMS, EXPECTED_TEAMS - resolved


def test_slate_date_is_not_host_utc_date():
    dt = datetime(2026, 8, 26, 2, 30, tzinfo=timezone.utc)
    assert slate_date(dt).isoformat() == '2026-08-25'


def test_doubleheader_odds_match_by_time():
    odds = [
        {'home_team':'New York Yankees','away_team':'Boston Red Sox','commence_time':'2026-08-25T17:05:00Z'},
        {'home_team':'New York Yankees','away_team':'Boston Red Sox','commence_time':'2026-08-25T23:05:00Z'},
    ]
    g1 = {'local':'New York Yankees','visita':'Boston Red Sox','start_time_utc':'2026-08-25T17:10:00Z'}
    g2 = {'local':'New York Yankees','visita':'Boston Red Sox','start_time_utc':'2026-08-25T23:10:00Z'}
    assert match_odds_game(odds, g1)['commence_time'].startswith('2026-08-25T17')
    assert match_odds_game(odds, g2)['commence_time'].startswith('2026-08-25T23')
    assert match_odds_game(odds[:1], g2) is None


def test_bookmaker_snapshot_is_coherent_and_prefers_complete_book():
    conv = lambda x: 2.0 if x == 100 else 1.91
    event = {
        'home_team':'New York Yankees','away_team':'Boston Red Sox',
        'bookmakers':[
            {'key':'thin','markets':[{'key':'h2h','outcomes':[{'name':'New York Yankees','price':100},{'name':'Boston Red Sox','price':100}]}]},
            {'key':'full','markets':[
                {'key':'h2h','outcomes':[{'name':'New York Yankees','price':100},{'name':'Boston Red Sox','price':100}]},
                {'key':'totals','outcomes':[{'name':'Over','point':8.5,'price':-110},{'name':'Under','point':8.5,'price':-110}]},
                {'key':'spreads','outcomes':[{'name':'New York Yankees','point':-1.5,'price':-110},{'name':'Boston Red Sox','point':1.5,'price':-110}]},
            ]},
        ]
    }
    snap = market_from_event(event, conv)
    assert snap['bookmaker'] == 'full'
    assert snap['linea_carreras'] == 8.5
    assert snap['spread_loc'] == -1.5 and snap['spread_vis'] == 1.5


def test_auto_weather_is_conservative():
    now = datetime(2026, 8, 25, 16, 0, tzinfo=timezone.utc)
    assert conservative_auto_weather('Arizona Diamondbacks', '2026-08-25T17:00:00Z', 105, 15, 'Outfield', now)[3] == 'neutral_roof_unknown'
    assert conservative_auto_weather('Boston Red Sox', '2026-08-25T23:00:00Z', 92, 18, 'Outfield', now)[3] == 'neutral_not_near_first_pitch'


def test_push_aware_ev():
    c = total_candidate('Over 8.0', 60, 60, 2.0, 0.50, prob_push_mc=10.0)
    assert c is not None
    assert c.push_probability == 5.0
    assert abs(c.ev_pct - 25.0) < 1e-9
    assert abs(c.edge_pp - (60/95*100 - 50)) < 0.01


def test_monte_carlo_reproducible_and_spread_pushes_present():
    kwargs = dict(
        local='NYY', visita='BOS', pitcher_loc_xfip=4.0, pitcher_vis_xfip=4.2,
        wrc_loc=102, wrc_vis=99, bullpen_loc_era=3.9, bullpen_vis_era=4.1,
        park_factor=102, altitud_ft=54, viento_mph=0, direccion_viento='None', temp_f=72,
        linea_carreras_casino=8.0, df_games=pd.DataFrame(), num_simulaciones=5000,
    )
    a = simular_partido_mlb(**kwargs)
    b = simular_partido_mlb(**kwargs)
    assert a == b
    assert 'Push 8.0' in a['Carreras']
    assert 'Push Spread Local +2.0' in a['Carreras']
    assert a['Metadatos']['Simulation_Seed'] == b['Metadatos']['Simulation_Seed']


def test_cache_signature_detects_middle_row_changes():
    a = pd.DataFrame({'Team':['A','B','C'], 'Season':[2025,2025,2025], 'OPS_Index':[90,100,110]})
    b = a.copy(); b.loc[1, 'OPS_Index'] = 101
    assert _frame_signature(a, ('Team','Season','OPS_Index')) != _frame_signature(b, ('Team','Season','OPS_Index'))


def test_partial_game_type_preserves_legacy_history():
    df = pd.DataFrame([
        {'Date':'2025-04-01','Home':'NYY','Away':'BOS','Home_Score':5,'Away_Score':3,'Season':2025,'GameType':None},
        {'Date':'2026-04-01','Home':'NYY','Away':'BOS','Home_Score':4,'Away_Score':2,'Season':2026,'GameType':'R'},
        {'Date':'2026-03-10','Home':'NYY','Away':'BOS','Home_Score':7,'Away_Score':6,'Season':2026,'GameType':None},
        {'Date':'2026-03-20','Home':'NYY','Away':'BOS','Home_Score':7,'Away_Score':6,'Season':2026,'GameType':'S'},
    ])
    g = prepare_games(df)
    assert len(g) == 2
    assert set(g['Season'].astype(int)) == {2025, 2026}


def test_settlement_win_loss_push():
    game = pd.Series({'Home_Score':5, 'Away_Score':3})
    over_push = pd.Series({'market':'Totales','selection':'Over 8.0','line':8.0,'odds':1.91,'home':'H','away':'A'})
    assert settle_row(over_push, game)[0] == 'push'
    home_ml = pd.Series({'market':'Moneyline','selection':'Gana Local (H)','line':None,'odds':1.80,'home':'H','away':'A'})
    status, profit, _ = settle_row(home_ml, game)
    assert status == 'win' and abs(profit - 0.8) < 1e-9


def test_security_and_workflow_guards():
    app = Path('app_mlb.py').read_text(encoding='utf-8')
    assert 'f9ffe1d7530a88b08e853659466c46ff' not in app
    assert 'match_odds_game' in app and 'game_pk' in app
    manual = Path('.github/workflows/actualizar_pitchers.yml').read_text(encoding='utf-8')
    daily = Path('.github/workflows/update_mlb.yml').read_text(encoding='utf-8')
    assert 'schedule:' not in manual
    assert 'concurrency:' in daily and 'python settle_picks.py' in daily


def main():
    tests = [v for k,v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for fn in tests:
        fn()
        print('PASS', fn.__name__)
    print(f'V5 integrity tests passed: {len(tests)}')


if __name__ == '__main__':
    main()
