import pandas as pd

from modules.bigdata_mlb import MLBDataWarehouse


def _games():
    return pd.DataFrame([
        {"Date":"2025-04-01","Season":2025,"GameType":"R","Home":"NYY","Away":"BOS","Home_Score":5,"Away_Score":3},
        {"Date":"2025-04-02","Season":2025,"GameType":"R","Home":"BOS","Away":"NYY","Home_Score":4,"Away_Score":2},
        {"Date":"2025-04-03","Season":2025,"GameType":"R","Home":"NYY","Away":"BOS","Home_Score":7,"Away_Score":1},
    ])


def test_first_game_uses_priors_only():
    f = MLBDataWarehouse.build_leak_safe_features(_games())
    first = f.iloc[0]
    assert first.home_win5 == 0.5
    assert first.away_win5 == 0.5
    assert first.h2h_home_win == 0.5


def test_future_result_does_not_change_past_features():
    g = _games()
    a = MLBDataWarehouse.build_leak_safe_features(g)
    g.loc[2, "Home_Score"] = 30
    g.loc[2, "Away_Score"] = 0
    b = MLBDataWarehouse.build_leak_safe_features(g)
    cols = [c for c in a.columns if c.startswith("home_") or c.startswith("away_") or c.startswith("h2h_")]
    pd.testing.assert_series_equal(a.loc[1, cols], b.loc[1, cols])


def test_previous_day_updates_next_game():
    f = MLBDataWarehouse.build_leak_safe_features(_games())
    second = f.iloc[1]
    assert second.home_win5 == 0.0
    assert second.away_win5 == 1.0


def test_same_day_result_never_updates_another_same_day_pregame_row():
    doubleheader = pd.DataFrame([
        {"GameID":1001,"Date":"2025-06-10","Season":2025,"GameType":"R","Home":"NYY","Away":"BOS","Home_Score":10,"Away_Score":0},
        {"GameID":1002,"Date":"2025-06-10","Season":2025,"GameType":"R","Home":"BOS","Away":"NYY","Home_Score":1,"Away_Score":2},
    ])
    f = MLBDataWarehouse.build_leak_safe_features(doubleheader)
    assert len(f) == 2
    # Both games are evaluated from the same prior-day state, regardless of GameID order.
    assert f.iloc[0].home_win5 == 0.5 and f.iloc[0].away_win5 == 0.5
    assert f.iloc[1].home_win5 == 0.5 and f.iloc[1].away_win5 == 0.5
    assert f.iloc[0].h2h_home_win == 0.5 and f.iloc[1].h2h_home_win == 0.5


if __name__ == '__main__':
    test_first_game_uses_priors_only()
    test_future_result_does_not_change_past_features()
    test_previous_day_updates_next_game()
    test_same_day_result_never_updates_another_same_day_pregame_row()
    print('Big Data leak-safety: OK')
