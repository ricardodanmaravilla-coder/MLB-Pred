from types import SimpleNamespace

from modules.pick_ledger import enrich_tracking_row
from settle_picks import settle_row


def _game(home, away):
    return SimpleNamespace(Home_Score=home, Away_Score=away)


def test_moneyline_win_profit_mxn():
    row = enrich_tracking_row({
        'market':'Moneyline','selection':'Gana Local (NYY)','home':'NYY','away':'BOS',
        'odds':1.90,'prob_combined':60.0,'ev_pct':14.0,
    })
    result, units, score = settle_row(row, _game(5, 3))
    assert result == 'win' and units == 0.9 and score == '5-3'
    assert round(row['stake_mxn'] * units, 2) > 0


def test_moneyline_loss_is_negative_stake():
    row = enrich_tracking_row({
        'market':'Moneyline','selection':'Gana Local (NYY)','home':'NYY','away':'BOS',
        'odds':1.90,'prob_combined':60.0,'ev_pct':14.0,
    })
    result, units, _ = settle_row(row, _game(2, 4))
    assert result == 'loss' and units == -1.0
    assert round(row['stake_mxn'] * units, 2) == -row['stake_mxn']


def test_total_integer_push_has_zero_profit():
    row = enrich_tracking_row({
        'market':'Totales','selection':'Over 8.0','line':8.0,'odds':2.0,
        'prob_combined':60.0,'ev_pct':25.0,
    })
    result, units, _ = settle_row(row, _game(5, 3))
    assert result == 'push' and units == 0.0
    assert row['kelly_pct'] > 0


def test_runline_settlement():
    row = enrich_tracking_row({
        'market':'Hándicap','selection':'Hándicap -1.5 (NYY)','home':'NYY','away':'BOS',
        'line':-1.5,'odds':1.95,'prob_combined':61.0,'ev_pct':18.95,
    })
    assert settle_row(row, _game(6, 3))[0] == 'win'
    assert settle_row(row, _game(5, 4))[0] == 'loss'


def main():
    tests=[v for k,v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for fn in tests:
        fn(); print('PASS', fn.__name__)
    print(f'Settlement tracking tests passed: {len(tests)}')


if __name__ == '__main__':
    main()
